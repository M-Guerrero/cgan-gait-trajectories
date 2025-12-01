import argparse, os, numpy as np, h5py
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from train_cgan import Generador


# ------------------------------------------------------------
# Carga de datos en formato HDF5 compatible con h5py
# ------------------------------------------------------------

def load(path: str):
    """
    Carga un archivo HDF5 (.mat) con las tres variables requeridas:

        - X_cond : matriz de condiciones (N, 2)
        - Y_seq  : secuencias articulares (N, T, J)
        - S_meas : medidas antropométricas u otras características (N, 6)

    La función verifica que el número de muestras N coincide en las tres matrices.
    """
    with h5py.File(path, 'r') as f:
        X = np.array(f['X_cond'], dtype=np.float32)   # (N, 2)
        Y = np.array(f['Y_seq'],  dtype=np.float32)   # (N, T, J)
        S = np.array(f['S_meas'], dtype=np.float32)   # (N, 6)

    if not (X.shape[0] == Y.shape[0] == S.shape[0]):
        raise ValueError(f"Error de shape: X={X.shape}, Y={Y.shape}, S={S.shape}")

    return {'X': X, 'Y': Y, 'S': S}


# ------------------------------------------------------------
# Visualización de secuencias sintéticas y tabla de medidas
# ------------------------------------------------------------

def plot_con_tabla(y, joints, Lm, Hm, s_row, out_png, titulo=""):
    """
    Genera una figura compuesta por:

      - Curvas articulares a lo largo del ciclo de marcha.
      - Una tabla con las condiciones de longitud y altura de paso,
        así como las medidas antropométricas asociadas al sujeto.

    Parámetros
    ----------
    y : np.ndarray
        Secuencia articular de tamaño (T, J).
    joints : list of str
        Etiquetas de las articulaciones en el mismo orden que las columnas de y.
    Lm, Hm : float
        Longitud y altura de paso en metros.
    s_row : np.ndarray o None
        Vector de medidas antropométricas (dimensión 6) para el sujeto.
    out_png : str
        Ruta donde se almacena la figura exportada en formato PNG.
    titulo : str
        Título opcional que se muestra sobre las curvas articulares.
    """
    T_, J = y.shape
    t = np.linspace(0, 100, T_, dtype=np.float32)  # eje temporal en porcentaje del ciclo

    fig = plt.figure(figsize=(12, 6), dpi=150)
    gs = GridSpec(1, 3, width_ratios=[2, 0.05, 1], wspace=0.3)

    # Subgráfico de curvas articulares
    ax = fig.add_subplot(gs[0, 0])
    for j in range(J):
        lbl = joints[j] if (joints and j < len(joints)) else f'J{j+1}'
        ax.plot(t, y[:, j], label=lbl)

    if titulo:
        ax.set_title(titulo)

    ax.set_xlabel('% ciclo')
    ax.set_ylabel('ángulo (°)')
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.2)

    # Columna intermedia vacía (funciona como separador visual)
    fig.add_subplot(gs[0, 1]).axis('off')

    # Subgráfico con tabla de medidas
    ax_tab = fig.add_subplot(gs[0, 2])
    ax_tab.axis('off')

    # Nombres de las medidas representadas
    meas_names = [
        'stepLength_m', 'stepHeight_m',
        'Height', 'femur', 'tibia', 'Leg_length', 'KneeWidth', 'widthPelvis'
    ]

    # Filas iniciales: longitud y altura de paso
    rows = [
        [meas_names[0], f"{Lm:.3f}"],
        [meas_names[1], f"{Hm:.3f}"]
    ]

    # Inclusión de medidas antropométricas si están disponibles
    if s_row is not None and len(s_row) == 6:
        for name, val in zip(meas_names[2:], s_row):
            rows.append([name, f"{float(val):.3f}"])

    # Construcción de la tabla en el eje correspondiente
    tbl = ax_tab.table(
        cellText=rows,
        colLabels=["Measure (m)", "Value"],
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    # Escalado vertical de la tabla según el número de filas
    tbl.scale(1.0, max(1.6, len(rows) / 9))

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# ------------------------------------------------------------
# Guardado de muestras sintéticas en formato HDF5 compatible
# ------------------------------------------------------------

def save_mat(path, Y_mat, X_mat, S_mat):
    """
    Almacena las matrices generadas en un archivo HDF5 (.mat)
    con la estructura esperada por los scripts de evaluación:

        - Y_seq  : (N, T, 4)
        - X_cond : (N, 2)
        - S_meas : (N, 6)

    Este formato es compatible con la lectura mediante h5py:

        with h5py.File(path, 'r') as f:
            X = f['X_cond'][...]
            Y = f['Y_seq'][...]
            S = f['S_meas'][...]

    Parámetros
    ----------
    path : str
        Ruta del archivo de salida.
    Y_mat : np.ndarray
        Secuencias articulares generadas (N, T, 4).
    X_mat : np.ndarray
        Condiciones de longitud y altura de paso (N, 2).
    S_mat : np.ndarray
        Medidas antropométricas asociadas (N, 6).
    """
    with h5py.File(path, 'w') as f:
        f.create_dataset('Y_seq',  data=Y_mat.astype(np.float32))
        f.create_dataset('X_cond', data=X_mat.astype(np.float32))
        f.create_dataset('S_meas', data=S_mat.astype(np.float32))


# ------------------------------------------------------------
# Rutina principal de generación de muestras sintéticas
# ------------------------------------------------------------

def main(args):
    """
    Procedimiento principal de generación de muestras sintéticas a partir
    de un generador condicional previamente entrenado.

    El script realiza las siguientes operaciones:

        1. Carga el checkpoint del modelo (incluyendo parámetros y normalización).
        2. Reconstruye la arquitectura del generador `Generador`.
        3. Carga las condiciones reales (L, H) y las medidas antropométricas S.
        4. Normaliza las medidas S y construye el vector de condición completo.
        5. Genera un conjunto de secuencias articulares sintéticas en modo batch.
        6. Desnormaliza las secuencias según las estadísticas almacenadas.
        7. Guarda los resultados en un archivo HDF5 (.mat) compatible.
        8. Opcionalmente, produce figuras de vista previa con curvas y tabla de medidas.
    """
    # Verificación de disponibilidad de GPU
    assert torch.cuda.is_available(), "CUDA requerida"

    # Inicialización de semillas para reproducibilidad
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Selección de dispositivo de cálculo
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # Carga del checkpoint y reconstrucción del generador
    # --------------------------------------------------------
    ckpt = torch.load(args.ckpt, map_location='cpu')

    # Longitud temporal, número de articulaciones y dimensión condicional
    T = int(ckpt['T'])
    J = int(ckpt['J'])

    # La dimensión condicional se deduce a partir del primer peso lineal de la proyección
    C = ckpt['G']['cond_proj.mlp.0.weight'].shape[1]

    # Construcción del generador con la misma configuración utilizada en el entrenamiento
    G = Generador(
        X_dim=C,
        T=T,
        J=J,
        z_dim=16,
        hidden=128,
        X_feat=32,
        z_rho=0.97
    )
    G.load_state_dict(ckpt['G'])
    G.to(device).eval()

    # Escala del ruido latente (controla la dispersión de las muestras)
    G.z_dif(float(args.z))

    # Parámetros de normalización de las trayectorias articulares
    muY = ckpt['muY'].detach().cpu().numpy().astype(np.float32)
    sdY = ckpt['sdY'].detach().cpu().numpy().astype(np.float32)
    assert muY.shape == (1, 1, J) and sdY.shape == (1, 1, J)

    # --------------------------------------------------------
    # Carga de la base de datos de referencia
    # --------------------------------------------------------
    # El archivo .mat de entrada (por ejemplo, train_cont.mat o variantes)
    data = load(args.mat)
    X, S = data['X'], data['S']   # X: (N, 2), S: (N, 6)

    # ========================================================
    #   Modo PAIR: emparejamiento (L, H) con sus medidas S
    # ========================================================
    LH = X.astype(np.float32)  # condiciones clínicas originales (N_full, 2)

    # Selección opcional de un subconjunto de sujetos (si n > 0)
    if args.n and args.n > 0 and LH.shape[0] > args.n:
        idx = np.random.choice(LH.shape[0], size=args.n, replace=False)
    else:
        idx = np.arange(LH.shape[0])

    # Subconjunto de condiciones y medidas correspondientes
    LH = LH[idx]                              # (N_sel, 2)
    S_take = S[idx, :].astype(np.float32)     # (N_sel, 6)

    # --------------------------------------------------------
    # Normalización interna de las medidas S para el generador
    # --------------------------------------------------------
    mu = S.mean(axis=0, keepdims=True).astype(np.float32)
    sd = S.std(axis=0, keepdims=True).astype(np.float32)
    sd[sd == 0] = 1.0
    S_norm = (S_take - mu) / sd               # (N_sel, 6)

    # Vector de condición completo: [L, H] + medidas normalizadas
    conds_base = np.concatenate(
        [LH, S_norm],
        axis=1
    ).astype(np.float32)                      # (N_sel, C_condicional)

    # Repetición de condiciones para generar K muestras por condición
    conds = np.repeat(conds_base, args.K, axis=0).astype(np.float32)

    # Matrices de salida en el mismo orden que requiere el formato HDF5:
    #   - X_save: solo L y H, repetidos K veces.
    #   - S_raw : medidas originales (no normalizadas), repetidas K veces.
    X_save = np.repeat(LH, args.K, axis=0).astype(np.float32)      # (N, 2)
    S_raw  = np.repeat(S_take, args.K, axis=0).astype(np.float32)  # (N, 6)

    # Comprobación de consistencia dimensional con el generador
    assert conds.shape[1] == C, "Dimensión de conds no coincide con X_dim del generador"

    # --------------------------------------------------------
    # Generación de secuencias articulares sintéticas
    # --------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)

    N = conds.shape[0]
    Y = np.zeros((N, T, J), dtype=np.float32)

    # Generación en bloques para optimizar el uso de GPU
    with torch.no_grad():
        bs = min(1024, max(64, N))  # tamaño de batch adaptativo
        for i in range(0, N, bs):
            cc = torch.from_numpy(conds[i:i + bs]).to(device)
            y = G(cc).detach().cpu().numpy().astype(np.float32)  # (bs, T, J)

            # Desnormalización mediante los parámetros almacenados en el checkpoint
            y = y * sdY + muY
            Y[i:i + bs] = y

    # --------------------------------------------------------
    # Almacenamiento de las secuencias generadas en archivo .mat
    # --------------------------------------------------------
    out_mat = os.path.join(args.out, args.name)
    save_mat(out_mat, Y, X_save, S_raw)
    print("Archivo .mat (HDF5):", out_mat, Y.shape)

    # --------------------------------------------------------
    # Generación opcional de vistas previas (figuras PNG)
    # --------------------------------------------------------
    if args.previews > 0:
        joints = ['Hip_flex', 'Knee_flex', 'Ankle_flex', 'Pelvis_List']
        pick = np.linspace(0, N - 1, min(args.previews, N), dtype=int)

        for k, ix in enumerate(pick):
            Lm, Hm = float(X_save[ix, 0]), float(X_save[ix, 1])
            s_row = S_raw[ix]
            titulo = f"L={Lm:.3f} m | H={Hm:.3f} m"
            out_png = os.path.join(args.out, f'preview_{k:02d}.png')

            plot_con_tabla(Y[ix], joints, Lm, Hm, s_row, out_png, titulo)
            print("fig:", out_png)


# ------------------------------------------------------------
# Punto de entrada del script
# ------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)                     # ruta al checkpoint del modelo entrenado
    ap.add_argument('--mat', default='train_db_original.mat')    # archivo .mat con condiciones reales
    ap.add_argument('--out', default='samples')                  # carpeta de salida para muestras y figuras
    ap.add_argument('--K', type=int, default=1)                  # número de muestras por condición
    ap.add_argument('--n', type=int, default=0)                  # número de sujetos a seleccionar (0 = todos)
    ap.add_argument('--previews', type=int, default=6)           # número máximo de figuras de vista previa
    ap.add_argument('--seed', type=int, default=123)             # semilla para reproducibilidad
    ap.add_argument('--z', type=float, default=1.0)              # escala del ruido latente
    ap.add_argument('--name', default='synthetic_samples.mat')   # nombre del archivo .mat de salida
    args = ap.parse_args()
    main(args)
