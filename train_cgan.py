import argparse, os, time, math, numpy as np, h5py
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ------------------------------------------------------------
# Funciones de carga de datos
# ------------------------------------------------------------

def load(path: str):
    """
    Carga los datos de un archivo .mat compatible con el formato esperado:
        - X_cond: matriz de condiciones clínicas (N, C)
        - Y_seq : secuencias articulares (N, T, J)
        - S_meas: medidas antropométricas u otras características del sujeto (N, 6)

    La función verifica que el número de muestras N coincide en las tres variables.
    """
    with h5py.File(path, 'r') as f:
        X = np.array(f['X_cond'], dtype=np.float32)   # (N, C)
        Y = np.array(f['Y_seq'],  dtype=np.float32)   # (N, T, J)
        S = np.array(f['S_meas'], dtype=np.float32)   # (N, 6)

    if not (X.shape[0] == Y.shape[0] == S.shape[0]):
        raise ValueError(f"Error de shape: X={X.shape}, Y={Y.shape}, S={S.shape}")

    return {'X': X, 'Y': Y, 'S': S}


# ------------------------------------------------------------
# Dataset para el entrenamiento del modelo generativo
# ------------------------------------------------------------

class GANDataset(Dataset):
    """
    Conjunto de datos para el entrenamiento del modelo generativo.

    - Normaliza las condiciones clínicas (L, H) y las medidas del sujeto S.
    - Normaliza las trayectorias Y de forma global en tiempo y articulaciones.
    - Almacena las medias y desviaciones típicas para poder desnormalizar más adelante.
    """

    def __init__(self, path="train_cont.mat", usar_meas=True):
        # Carga datos brutos
        d = load(path)
        X, Y, S = d['X'], d['Y'], d['S']

        # Extrae longitud y altura de paso (primeras dos columnas de X_cond)
        LH = X[:, :2]

        # Normalización de L y H (media y desviación típicas globales)
        m_LH = LH.mean(axis=0, keepdims=True)
        s_LH = LH.std(axis=0, keepdims=True)
        s_LH[s_LH == 0] = 1.0  # Evita divisiones por cero
        LHn = (LH - m_LH) / s_LH

        # Normalización de medidas del sujeto, si se utilizan
        if usar_meas:
            m_S = S.mean(axis=0, keepdims=True)
            s_S = S.std(axis=0, keepdims=True)
            s_S[s_S == 0] = 1.0
            Sn = (S - m_S) / s_S

            # Condición completa: [L,H] normalizados + medidas del sujeto normalizadas
            X_full = np.concatenate([LHn, Sn], axis=1)
        else:
            m_S = None
            s_S = None
            X_full = LHn

        # Normalización de las trayectorias Y sobre todas las muestras y tiempos
        muY = Y.mean(axis=(0, 1), keepdims=True)   # media global por articulación
        sdY = Y.std(axis=(0, 1), keepdims=True)    # desviación típica por articulación
        sdY[sdY == 0] = 1.0
        Yn = (Y - muY) / sdY

        # Se almacenan los parámetros de normalización como tensores
        self.muY = torch.from_numpy(muY.astype(np.float32))
        self.sdY = torch.from_numpy(sdY.astype(np.float32))
        self.m_LH = torch.from_numpy(m_LH.astype(np.float32))
        self.s_LH = torch.from_numpy(s_LH.astype(np.float32))

        if m_S is not None:
            self.m_S = torch.from_numpy(m_S.astype(np.float32))
            self.s_S = torch.from_numpy(s_S.astype(np.float32))
        else:
            self.m_S = None
            self.s_S = None

        # Conversión a tensores de PyTorch
        self.X = torch.from_numpy(X_full.copy())  # Condiciones normalizadas
        self.Y = torch.from_numpy(Yn.copy())      # Trayectorias normalizadas

        # Dimensiones principales del dataset
        self.N, self.T, self.J = self.Y.shape
        self.C = self.X.shape[1]

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        """
        Devuelve el par (condición, trayectoria) para el índice i.
        """
        return self.X[i], self.Y[i]


# ------------------------------------------------------------
# Bloques básicos de red (RNN, proyección condicional, cabezas por canal)
# ------------------------------------------------------------

class BloqueRNN(nn.Module):
    """
    Bloque recurrente basado en GRU.

    Permite configurar el tamaño de la entrada, el tamaño oculto y si la red
    es bidireccional o no.
    """

    def __init__(self, tam_entrada, tam_oculto, bidireccional=False):
        super().__init__()
        self.rnn = nn.GRU(
            tam_entrada,
            tam_oculto,
            batch_first=True,
            bidirectional=bidireccional
        )

    def forward(self, x_o, h_o=None):
        return self.rnn(x_o, h_o)


class ProyCond(nn.Module):
    """
    Proyección de las condiciones clínicas a un espacio latente temporal.

    La condición (vector de características) se transforma mediante un MLP
    y, posteriormente, se replica a lo largo del eje temporal para poder
    concatenarse con las secuencias generadas o reales.
    """

    def __init__(self, dim_cond, dim_out):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_cond, max(16, dim_out)),
            nn.SiLU(),
            nn.Linear(max(16, dim_out), dim_out)
        )

        # Inicialización de pesos lineales con Xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, C, T):
        """
        Proyecta el vector de condición C a dimensión dim_out
        y lo expande a una secuencia de longitud T.

        Entrada:
            C: (B, dim_cond)
            T: número de instantes temporales
        Salida:
            (B, T, dim_out)
        """
        f = self.mlp(C)
        return f.view(C.size(0), 1, -1).expand(-1, T, -1)


class CabezasPorCanal(nn.Module):
    """
    Módulo de salida con una "cabeza" independiente por articulación.

    Cada articulación se modela con un pequeño MLP que recibe el estado oculto
    y produce una señal escalar, lo que permite cierta especialización por canal.
    """

    def __init__(self, h, J, width_ratio=0.5):
        super().__init__()
        mid = max(8, int(h * width_ratio))
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(h, mid),
                nn.SiLU(),
                nn.Linear(mid, 1)
            )
            for _ in range(J)
        ])

    def forward(self, x):
        """
        Entrada:
            x: (B, T, h)
        Salida:
            (B, T, J)
        """
        y = [mlp(x) for mlp in self.mlps]
        return torch.cat(y, dim=-1)


# ------------------------------------------------------------
# Generador
# ------------------------------------------------------------

class Generador(nn.Module):
    """
    Generador condicional basado en ruido AR(1) y bloques GRU.

    - Recibe como entrada un vector de condición C (L, H y medidas del sujeto).
    - Genera una secuencia de ruido temporalmente correlacionado (proceso AR(1)).
    - Concatena el ruido con la proyección de la condición.
    - Pasa la secuencia resultante por dos capas GRU y normalización de capas.
    - Produce como salida una secuencia temporal (B, T, J) con una cabeza por articulación.
    """

    def __init__(self, X_dim=2, T=100, J=4, z_dim=16, hidden=128, X_feat=32, z_rho=0.97):
        super().__init__()
        self.T = T
        self.J = J
        self.z_dim = z_dim

        # Correlación temporal del ruido
        self.z_rho = float(z_rho)
        self.z_mult = 1.0  # factor de escala ajustable para el ruido

        # Proyección de condiciones
        self.cond_proj = ProyCond(X_dim, X_feat)

        # Bloques recurrentes
        self.rnn1 = BloqueRNN(z_dim + X_feat, hidden, bidireccional=False)
        self.rnn2 = BloqueRNN(hidden, hidden, bidireccional=False)

        # Normalización y regularización
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(0.1)

        # Cabezas independientes por articulación
        self.head = CabezasPorCanal(hidden, J, width_ratio=0.5)

        # Inicialización de capas lineales
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def z_dif(self, s: float):
        """
        Ajusta el factor de escala del ruido latente.
        Este parámetro permite modificar la dispersión de las secuencias generadas.
        """
        self.z_mult = float(s)

    def ar1_z(self, B, T):
        """
        Genera una secuencia de ruido latente con estructura AR(1).

        Entrada:
            B: tamaño de batch
            T: longitud temporal
        Salida:
            z: tensor de ruido (B, T, z_dim) con correlación temporal controlada por z_rho.
        """
        dev = next(self.parameters()).device
        rho = self.z_rho

        # Ruido base independiente
        z0 = torch.randn(B, T, self.z_dim, device=dev) * self.z_mult

        # Secuencia AR(1)
        z = torch.empty_like(z0)
        z[:, 0] = z0[:, 0]
        c = math.sqrt(max(0.0, 1.0 - rho * rho))
        for t in range(1, T):
            z[:, t] = rho * z[:, t - 1] + c * z0[:, t]

        return z

    def forward(self, cond, z=None):
        """
        Generación de secuencias condicionales.

        Entrada:
            cond: tensor de condiciones (B, X_dim)
            z   : opcional, ruido externo (B, T, z_dim).
        Salida:
            y   : secuencia generada (B, T, J)
        """
        B = cond.size(0)
        T = self.T

        # Si no se proporciona ruido explícito, se genera uno AR(1)
        if z is None:
            z = self.ar1_z(B, T)

        # Concatenación de ruido y condición proyectada
        x, _ = self.rnn1(torch.cat([z, self.cond_proj(cond, T)], dim=-1))
        x = self.ln1(x)
        x = self.drop(x)

        # Segundo bloque recurrente y normalización
        x, _ = self.rnn2(x)
        x = self.ln2(x)

        # Proyección final a las articulaciones
        return self.head(x)


# ------------------------------------------------------------
# Discriminador
# ------------------------------------------------------------

class Discriminador(nn.Module):
    """
    Discriminador condicional con estructura bidireccional.

    - Recibe secuencias articulares y las condiciones asociadas.
    - Combina ambas entradas mediante una proyección de la condición en el dominio temporal.
    - Procesa la secuencia con una GRU bidireccional.
    - Agrega la información en el tiempo y produce una puntuación escalar de real/falso.
    - Devuelve también una representación intermedia (feat) útil para la pérdida de feature matching.
    """

    def __init__(self, cond_dim=2, T=100, J=4, hidden=128):
        super().__init__()

        # Proyección temporal de las condiciones
        self.C_x_T = ProyCond(cond_dim, 16)

        # Bloque recurrente bidireccional
        self.rnn = BloqueRNN(J + 16, hidden, bidireccional=True)

        # Cabeza de clasificación adversaria
        self.fc_adv = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1)
        )

        # Inicialización de capas lineales
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, cond, y):
        """
        Entrada:
            cond: condiciones (B, cond_dim)
            y   : secuencias articulares (B, T, J)
        Salida:
            s    : puntuación adversaria (B, 1)
            feat : representación agregada (B, 2*hidden)
        """
        B, T, _ = y.size()

        # Concatenación de secuencia y condición proyectada
        x, _ = self.rnn(torch.cat([y, self.C_x_T(cond, T)], dim=-1))

        # Agregación temporal mediante promedio
        feat = x.mean(dim=1)

        # Puntuación adversaria
        s = self.fc_adv(feat)
        return s, feat


# ------------------------------------------------------------
# Predictor de L y H a partir de trayectorias
# ------------------------------------------------------------

class PredictorLH(nn.Module):
    """
    Predictor de longitud y altura de paso a partir de las trayectorias articulares.

    - Emplea convoluciones 1D sobre el eje temporal para extraer características de la secuencia.
    - Agrega opcionalmente las medidas del sujeto S.
    - Devuelve una estimación de [L, H] normalizadas.
    """

    def __init__(self, J, S_dim, hid=64):
        super().__init__()

        # Extracción de características temporales mediante convoluciones
        self.fe = nn.Sequential(
            nn.Conv1d(J, 32, 5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.SiLU()
        )

        # Cabeza de regresión sobre [L, H]
        self.head = nn.Sequential(
            nn.Linear(64 + S_dim, hid),
            nn.SiLU(),
            nn.Linear(hid, 2)
        )

        # Inicialización de capas lineales y convolucionales
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, Y_lin, S_in):
        """
        Entrada:
            Y_lin : secuencia (B, T, J) normalizada
            S_in  : medidas del sujeto (B, S_dim) o None
        Salida:
            pred  : vector (B, 2) con la estimación de [L, H]
        """
        # Reordenación a formato (B, J, T) para convolución temporal
        x = Y_lin.permute(0, 2, 1)
        x = self.fe(x).mean(dim=2)  # agregación temporal por promedio

        # Concatenación con medidas del sujeto si se encuentran disponibles
        if S_in is not None and S_in.numel() > 0:
            x = torch.cat([x, S_in], dim=1)

        return self.head(x)


# ------------------------------------------------------------
# Bucle principal de entrenamiento
# ------------------------------------------------------------

def main(args):
    """
    Procedimiento principal de entrenamiento del modelo generativo condicional
    con un discriminador tipo WGAN-GP y un predictor auxiliar de [L, H].
    """

    # Comprobación de disponibilidad de GPU
    assert torch.cuda.is_available(), "CUDA requerida"
    device = torch.device('cuda')

    # "Warm-up" inicial para activar la GPU
    _ = torch.randn(1, device=device)
    torch.cuda.synchronize()

    # Fijación de semillas para reproducibilidad
    torch.manual_seed(args.seed or 0)
    np.random.seed(args.seed or 0)

    # Construcción del dataset y cálculo de dimensiones
    ds = GANDataset(args.mat, usar_meas=True)
    N, T, J = ds.N, ds.T, ds.J
    C = ds.C
    S_dim = max(0, C - 2)

    # Parámetros de normalización (trayectorias) a dispositivo
    muY = ds.muY.to(device)
    sdY = ds.sdY.to(device)

    # División entrenamiento/validación (10 % para validación)
    val_len = int(math.floor(0.10 * len(ds)))
    train_len = len(ds) - val_len
    tr, va = random_split(
        ds,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed or 0)
    )

    # DataLoaders para entrenamiento y validación
    dl_tr = DataLoader(
        tr, batch_size=64, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True
    )
    dl_va = DataLoader(
        va, batch_size=64, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=True
    )

    # Definición de modelos
    G = Generador(
        X_dim=C, T=T, J=J, z_dim=16,
        hidden=128, X_feat=32, z_rho=0.97
    ).to(device)

    D = Discriminador(
        cond_dim=C, T=T, J=J, hidden=128
    ).to(device)

    P = PredictorLH(
        J, S_dim=S_dim, hid=64
    ).to(device)

    # Optimizadores
    optG = torch.optim.Adam(G.parameters(), lr=1.0e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=8.0e-5, betas=(0.5, 0.999))
    optP = torch.optim.Adam(P.parameters(), lr=1.0e-3)

    # Planificadores de tasa de aprendizaje basados en el error de validación
    schG = ReduceLROnPlateau(optG, mode='min', factor=0.5, patience=10, verbose=False)
    schD = ReduceLROnPlateau(optD, mode='min', factor=0.5, patience=10, verbose=False)

    # Carpeta de salida
    os.makedirs(args.out, exist_ok=True)
    best = float('inf')
    best_path = os.path.join(args.out, 'ckpt_best.pt')

    # --------------------------------------------------------
    # Definición de funciones de pérdida auxiliares
    # --------------------------------------------------------

    def mse(a, b):
        """Error cuadrático medio global."""
        return ((a - b) ** 2).mean()

    def suavidad(y):
        """Penaliza diferencias entre muestras consecutivas (1ª derivada)."""
        return ((y[:, 1:, :] - y[:, :-1, :]) ** 2).mean()

    def curvatura(y):
        """Penaliza la segunda derivada discreta (curvatura temporal)."""
        return ((y[:, 2:, :] - 2 * y[:, 1:-1, :] + y[:, :-2, :]) ** 2).mean()

    def jerk(y):
        """Penaliza la tercera derivada discreta (jerk temporal)."""
        return (
            (y[:, 3:, :] - 3 * y[:, 2:-1, :] + 3 * y[:, 1:-2, :] - y[:, :-3, :]) ** 2
        ).mean()

    def cierre_ciclo(y):
        """Penaliza la diferencia entre el inicio y el final del ciclo."""
        return ((y[:, 0, :] - y[:, -1, :]) ** 2).mean()

    # Parámetros de WGAN-GP y control de entrenamiento
    n_critic = 3       # pasos del discriminador por actualización del generador
    gp_w = 10.0        # peso del término de gradiente penalty
    clip_grad = 10.0   # límite de norma de gradiente
    no_imp = 0         # contador de épocas sin mejora

    # --------------------------------------------------------
    # Bucle de entrenamiento por épocas
    # --------------------------------------------------------
    for ep in range(1, args.epocas + 1):
        G.train()
        D.train()
        P.train()

        t0 = time.time()
        total = len(dl_tr)

        # ------------------------------
        # Bucle de entrenamiento (batch)
        # ------------------------------
        for b, (C_np, Y_np) in enumerate(dl_tr, start=1):
            Cb = C_np.to(device, non_blocking=True)
            Yb = Y_np.to(device, non_blocking=True)

            # Condiciones L y H normalizadas
            LH = torch.stack([Cb[:, 0], Cb[:, 1]], dim=1)

            # Medidas del sujeto normalizadas (si existen)
            S_in = Cb[:, 2:] if S_dim > 0 else None

            # ----------------------------------
            # Actualización del discriminador D
            # ----------------------------------
            for _ in range(n_critic):
                with torch.no_grad():
                    # Se generan trayectorias sintéticas de forma determinista
                    Yf_det = G(Cb)

                # Evaluación de datos reales y generados
                d_r, _ = D(Cb, Yb)
                d_f, _ = D(Cb, Yf_det)

                # Cálculo del gradiente penalty (WGAN-GP)
                eps = torch.rand(Yb.size(0), 1, 1, device=device)
                inter = (eps * Yb + (1 - eps) * Yf_det).requires_grad_(True)

                # Se desactiva temporalmente cudnn para la derivada respecto a inter
                with torch.backends.cudnn.flags(enabled=False):
                    d_i, _ = D(Cb, inter)

                grad = torch.autograd.grad(
                    d_i,
                    inter,
                    torch.ones_like(d_i),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]

                gp = ((grad.reshape(grad.size(0), -1).norm(2, dim=1) - 1.0) ** 2).mean()

                # Pérdida WGAN-GP
                lossD = -(d_r.mean() - d_f.mean()) + gp_w * gp

                optD.zero_grad(set_to_none=True)
                lossD.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=clip_grad)
                optD.step()

            # ----------------------------------
            # Actualización del predictor P
            # ----------------------------------
            pred_r = P(Yb, S_in)
            lossP = F.mse_loss(pred_r, LH)

            optP.zero_grad(set_to_none=True)
            lossP.backward()
            optP.step()

            # ----------------------------------
            # Actualización del generador G
            # ----------------------------------
            # Se congela P durante la actualización de G
            for p in P.parameters():
                p.requires_grad_(False)

            # Generación de nuevas muestras
            Yf = G(Cb)

            # Término adversario (WGAN, signo opuesto)
            d_f, f_fake = D(Cb, Yf)
            adv = 0.2 * (-d_f.mean())

            # Feature matching frente a características reales
            with torch.no_grad():
                _, f_real = D(Cb, Yb)
            fm = 0.9 * ((f_fake - f_real) ** 2).mean()

            # Términos de regularización biomecánica
            sm = 0.2 * suavidad(Yf)
            curv = 0.10 * curvatura(Yf)
            jrk = 0.04 * jerk(Yf)

            # Término de calentamiento (reconstrucción directa) en las primeras épocas
            warm = (1.0 * mse(Yf, Yb)) if ep <= 5 else torch.tensor(0.0, device=device)

            # Pérdida supervisada de reconstrucción (permanece activa)
            sup = 1.0 * mse(Yf, Yb)

            # Coherencia condicional: el predictor debe recuperar L y H a partir de Yf
            pred_f = P(Yf, S_in)
            L_loss = 0.4 * F.l1_loss(pred_f[:, 0], LH[:, 0])
            H_loss = 0.4 * F.l1_loss(pred_f[:, 1], LH[:, 1])

            # Penalización de cierre de ciclo
            cyc = 0.3 * cierre_ciclo(Yf)

            # Pérdida total del generador
            lossG = adv + fm + sm + curv + jrk + warm + sup + L_loss + H_loss + cyc

            optG.zero_grad(set_to_none=True)
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=clip_grad)
            optG.step()

            # Se desbloquean los parámetros de P
            for p in P.parameters():
                p.requires_grad_(True)

            # Impresión de métricas intermedias (cada 50 batches o en el primero)
            if (b % 50 == 0) or (b == 1):
                with torch.no_grad():
                    dL = (pred_f[:, 0] - LH[:, 0]).abs().mean().item()
                    dH = (pred_f[:, 1] - LH[:, 1]).abs().mean().item()
                print(
                    f"ep {ep} / b {b}/{total}  "
                    f"D={lossD.item():.3f}  G={lossG.item():.3f}  "
                    f"P={lossP.item():.3f}  dL={dL:.3f} dH={dH:.3f}"
                )

        # --------------------------------------------------------
        # Evaluación sobre el conjunto de validación
        # --------------------------------------------------------
        G.eval()
        D.eval()
        P.eval()

        with torch.no_grad():
            m = 0.0
            n = 0
            for C_np, Y_np in dl_va:
                Cb = C_np.to(device)
                Yb = Y_np.to(device)
                Yf = G(Cb)
                m += mse(Yf, Yb).item() * Cb.size(0)
                n += Cb.size(0)
            m /= max(1, n)

        # Actualización de planificadores de tasa de aprendizaje
        schG.step(m)
        schD.step(m)

        dt = time.time() - t0

        # --------------------------------------------------------
        # Guardado de checkpoints (último y mejor)
        # --------------------------------------------------------
        ckpt_last = {
            'epoch': ep,
            'val_mse': float(m),
            'best_val': float(best),
            'G': G.state_dict(),
            'D': D.state_dict(),
            'P': P.state_dict(),
            'T': T,
            'J': J,
            'cond_dim': C,
            'muY': muY.detach().cpu(),
            'sdY': sdY.detach().cpu(),
            'm_LH': ds.m_LH,   # (1,2) en CPU
            's_LH': ds.s_LH,
            'm_S': ds.m_S,     # puede ser None si usar_meas=False
            's_S': ds.s_S,
        }
        torch.save(ckpt_last, os.path.join(args.out, 'ckpt_last.pt'))

        if (m < (best - 1e-4)) or (best == float('inf')):
            prev = best
            best = m

            ckpt_best = {
                'epoch': ep,
                'val_mse': float(m),
                'best_val': float(best),
                'G': G.state_dict(),
                'D': D.state_dict(),
                'P': P.state_dict(),
                'T': T,
                'J': J,
                'cond_dim': C,
                'muY': muY.detach().cpu(),
                'sdY': sdY.detach().cpu(),
                'm_LH': ds.m_LH,
                's_LH': ds.s_LH,
                'm_S': ds.m_S,
                's_S': ds.s_S,
            }
            torch.save(ckpt_best, best_path)
            no_imp = 0

            if prev < float('inf'):
                print(f"[BEST] ep {ep}  {m:.5f}  (mejora {prev - m:.5f}) -> {best_path}")
            else:
                print(f"[BEST] ep {ep}  {m:.5f} -> {best_path}")
        else:
            no_imp += 1
            print(f"[no-improve {no_imp}] val_mse={m:.5f}  best={best:.5f}")
            if args.stop_best > 0 and no_imp >= args.stop_best:
                print(f"[early stop] {no_imp} val sin mejora. Fin")
                break

        print(
            f"ep {ep}, val_mse={m:.5f},  best={best:.5f},  "
            f"time={dt:.1f}s,  lrG={optG.param_groups[0]['lr']:.2e},  "
            f"lrD={optD.param_groups[0]['lr']:.2e}"
        )

    print("fin. best:", best_path)


# ------------------------------------------------------------
# Punto de entrada
# ------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument('--mat', default='train_cont.mat')              # ruta al archivo .mat de entrenamiento
    ap.add_argument('--out', default='runs/cgan')            # carpeta de salida para checkpoints
    ap.add_argument('--epocas', type=int, default=120)              # número máximo de épocas
    ap.add_argument('--seed', type=int, default=42)                 # semilla para reproducibilidad
    ap.add_argument('--stop_best', type=int, default=20)            # paciencia para parada temprana
    args = ap.parse_args()
    main(args)
