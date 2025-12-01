# eval_estadistica.py
# Evaluación compacta de las secuencias reales y sintéticas:
#   - Curvas promedio (μ ± 1σ) con RMSE y r anotados por articulación.
#   - Matriz de acoplamiento articular y su diferencia (real vs sintético).
#   - Análisis PCA conjunto con elipses de dispersión y centroides.
#   - Informe de texto con un resumen numérico de las métricas principales.

import argparse, os
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Nombres por defecto de las articulaciones (orden de columnas en Y_seq)
JOINTS = ['Hip_flex', 'Knee_flex', 'Ankle_flex', 'Pelvis_List']


# ------------------------------------------------------------
# I/O: carga de datos en formato HDF5 compatible
# ------------------------------------------------------------
def load(path):
    """
    Carga un archivo HDF5 (.mat) con la estructura:

        - Y_seq  : (N, T, J) o (J, T, N)
        - X_cond : (N, C) o (C, N)
        - S_meas : (N, K) o (K, N)

    La función:
        1) Asegura que Y_seq tenga tres dimensiones.
        2) Soporta tanto el formato (N, T, J) como (J, T, N), en cuyo caso
           transpone para obtener (N, T, J).
        3) Reordena X_cond y S_meas para que la primera dimensión sea N.
        4) Comprueba consistencia en el número de muestras N entre Y, X y S.
    """
    with h5py.File(path, 'r') as f:
        Y = np.array(f['Y_seq'], dtype=np.float32)
        X = np.array(f['X_cond'], dtype=np.float32)
        S = np.array(f['S_meas'], dtype=np.float32)

    if Y.ndim != 3:
        raise ValueError('Y_seq debe ser 3D')

    # Soporta ambos formatos: (N, T, J) o (J, T, N) con pocas articulaciones
    if Y.shape[0] <= 8 and Y.shape[0] < Y.shape[-1]:
        # Caso típico (J, T, N) → se pasa a (N, T, J)
        Y = np.transpose(Y, (2, 1, 0))

    # X y S se reordenan para que la dimensión 0 sea N
    if X.shape[0] < X.shape[-1]:
        X = X.T
    if S.shape[0] < S.shape[-1]:
        S = S.T

    # Verificación de consistencia en N
    if not (Y.shape[0] == X.shape[0] == S.shape[0]):
        raise ValueError(f'N desajustado: Y={Y.shape} X={X.shape} S={S.shape}')

    return Y, X, S


# ------------------------------------------------------------
# Métricas basadas en curvas promedio (μ-μ)
# ------------------------------------------------------------
def curve_mean_sd(Y):
    """
    Calcula la curva media y la desviación estándar temporal por articulación.

    Parámetros
    ----------
    Y : np.ndarray
        Tensor de dimensión (N, T, J).

    Devuelve
    --------
    Ym : np.ndarray
        Curva media (T, J).
    Sd : np.ndarray
        Desviación estándar (T, J).
    """
    return np.nanmean(Y, axis=0), np.nanstd(Y, axis=0)


def rmse_avgcurve(Ym1, Ym2):
    """
    Calcula el RMSE entre dos curvas medias Ym1 y Ym2 por articulación.

    Parámetros
    ----------
    Ym1, Ym2 : np.ndarray
        Tensores (T, J) con las curvas promedio.

    Devuelve
    --------
    rmse : np.ndarray
        RMSE por articulación (J,).
    """
    return np.sqrt(np.nanmean((Ym1 - Ym2) ** 2, axis=0))


def pearson_r_avgcurve(Ym1, Ym2):
    """
    Calcula el coeficiente de correlación de Pearson entre dos curvas medias.

    El cálculo se realiza de forma independiente por articulación, tratando
    cada serie temporal como una variable y el tiempo como eje de observaciones.

    Parámetros
    ----------
    Ym1, Ym2 : np.ndarray
        Tensores (T, J) con las curvas promedio.

    Devuelve
    --------
    r : np.ndarray
        Correlación por articulación (J,).
    """
    T, J = Ym1.shape
    r = np.zeros(J, np.float32)
    for j in range(J):
        a = Ym1[:, j] - np.nanmean(Ym1[:, j])
        b = Ym2[:, j] - np.nanmean(Ym2[:, j])
        den = (np.sqrt(np.nanvar(a)) * np.sqrt(np.nanvar(b)) + 1e-9)
        r[j] = float(np.nanmean(a * b) / den)
    return r


def joint_amp_from_mean(Ym):
    """
    Estima la amplitud fisiológica aproximada de cada articulación a partir
    de su curva media.

    La amplitud se define como el rango de la curva media:
        rango_j = max_t(mu_j(t)) - min_t(mu_j(t)).

    Parámetros
    ----------
    Ym : np.ndarray
        Curva media (T, J).

    Devuelve
    --------
    amp : np.ndarray
        Amplitud (rango) por articulación (J,).
    """
    return np.nanmax(Ym, axis=0) - np.nanmin(Ym, axis=0)


# ------------------------------------------------------------
# Coupling interarticular
# ------------------------------------------------------------
def _corr(x, y):
    """
    Correlación de Pearson entre dos vectores 1D con manejo de NaNs.

    Parámetros
    ----------
    x, y : np.ndarray
        Vectores a correlacionar (se aplanan internamente).

    Devuelve
    --------
    r : float
        Correlación escalar. Si hay pocos datos válidos, devuelve NaN.
    """
    x = x.reshape(-1)
    y = y.reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan
    x = x[m] - x[m].mean()
    y = y[m] - y[m].mean()
    den = (np.sqrt(x.var() + 1e-9) * np.sqrt(y.var() + 1e-9))
    return float((x * y).mean() / den)


def coupling_matrix(Y):
    """
    Construye la matriz de acoplamiento interarticular a partir de Y.

    Para cada muestra i y para cada par de articulaciones (a, b)
    se calcula la correlación temporal entre sus señales, y se promedia
    posteriormente sobre todo el conjunto.

    Parámetros
    ----------
    Y : np.ndarray
        Tensor (N, T, J) con las trayectorias articulares.

    Devuelve
    --------
    C : np.ndarray
        Matriz de acoplamiento promedio (J, J).
    """
    N, T, J = Y.shape
    C = np.zeros((J, J), np.float32)
    for i in range(N):
        Yi = Y[i]
        for a in range(J):
            for b in range(J):
                C[a, b] += _corr(Yi[:, a], Yi[:, b])
    return C / max(1, N)


def coupling_delta_scalar(Cr, Cs, mode="mean_abs"):
    """
    Resume la diferencia entre dos matrices de acoplamiento (real y sintética)
    en un escalar.

    Parámetros
    ----------
    Cr, Cs : np.ndarray
        Matrices de acoplamiento reales y sintéticas (J, J).
    mode : str
        'mean_abs'  → media del valor absoluto de la diferencia.
        'fro_norm'  → norma de Frobenius dividida por J^2.

    Devuelve
    --------
    delta : float
        Medida escalar de diferencia de acoplamiento.
    """
    D = Cs - Cr
    J = D.shape[0]
    if mode == "mean_abs":
        return float(np.mean(np.abs(D)))
    elif mode == "fro_norm":
        return float(np.linalg.norm(D, ord='fro') / (J * J))
    else:
        raise ValueError("mode desconocido")


# ------------------------------------------------------------
# PCA: solapamiento entre distribuciones real y sintética
# ------------------------------------------------------------
def pca_overlap_svd(Yr, Ys):
    """
    Aplica PCA mediante SVD sobre el conjunto combinado (real + sintético),
    proyectando cada muestra en el espacio de las dos primeras componentes.

    El procedimiento es:
        1) Reorganizar cada muestra como un vector plano (T*J).
        2) Apilar muestras reales y sintéticas.
        3) Estandarizar cada característica (media 0, varianza 1).
        4) Aplicar SVD y obtener las dos primeras componentes principales.
        5) Calcular la varianza explicada por PC1+PC2 y la distancia entre
           centroides (en el espacio de PC1-PC2).

    Parámetros
    ----------
    Yr, Ys : np.ndarray
        Tensores (N_r, T, J) y (N_s, T, J).

    Devuelve
    --------
    Zr, Zs : np.ndarray
        Coordenadas en PC1-PC2 para muestras reales y sintéticas.
    var_sum : float
        Proporción de varianza explicada por las dos primeras componentes.
    cent : float
        Distancia euclídea entre los centroides de ambas nubes en PC1-PC2.
    """
    Nr, Tr, Jr = Yr.shape
    Ns, Ts, Js = Ys.shape
    if (Tr != Ts) or (Jr != Js):
        raise ValueError('T y J deben coincidir para PCA')

    # Reorganización a vectores (T*J)
    Xr = Yr.reshape(Nr, Tr * Jr)
    Xs = Ys.reshape(Ns, Ts * Js)

    # Combinación y estandarización global
    X = np.vstack([Xr, Xs]).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)

    # SVD sobre datos estandarizados
    U, S, _ = np.linalg.svd(X, full_matrices=False)

    # Proyección en las dos primeras componentes
    Z = U[:, :2] * S[:2]
    Zr = Z[:Nr]
    Zs = Z[Nr:]

    # Centroides y varianza explicada
    c_r = Zr.mean(0)
    c_s = Zs.mean(0)
    cent = float(np.linalg.norm(c_r - c_s))
    var_sum = float((S[:2] ** 2).sum() / (S ** 2).sum())

    return Zr, Zs, var_sum, cent


# ------------------------------------------------------------
# Funciones auxiliares de plotting
# ------------------------------------------------------------
def _joint_labels(J):
    """
    Devuelve una lista de etiquetas para las articulaciones en orden.

    Si existen nombres definidos en JOINTS se utilizan, en caso contrario
    se generan etiquetas genéricas J1, J2, ...
    """
    return [JOINTS[i] if i < len(JOINTS) else f'J{i+1}' for i in range(J)]


def _annotate_cells(ax, M, fmt='{:.2f}'):
    """
    Anota cada celda de una matriz M sobre el eje ax con el formato dado.
    """
    J = M.shape[0]
    for i in range(J):
        for j in range(J):
            val = M[i, j]
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val),
                        ha='center', va='center', fontsize=8)


def save_coupling_heatmap(C, outpath, title,
                          vmin=-1, vmax=1, annotate_fmt='{:.2f}'):
    """
    Representa y guarda un mapa de calor de la matriz de acoplamiento C.

    La figura incluye:
        - Escala de color entre vmin y vmax.
        - Etiquetas de articulaciones en ambos ejes.
        - Anotación numérica de las celdas.
    """
    J = C.shape[0]
    plt.figure(figsize=(4.2 + 0.25 * J, 3.8 + 0.25 * J), dpi=140)
    im = plt.imshow(C, vmin=vmin, vmax=vmax, cmap='coolwarm')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = _joint_labels(J)
    plt.xticks(range(J), ticks, rotation=45, ha='right')
    plt.yticks(range(J), ticks)
    _annotate_cells(plt.gca(), C, annotate_fmt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_coupling_delta(Cr, Cs, outpath, annotate_fmt='{:+.2f}'):
    """
    Construye y guarda la matriz de diferencias de acoplamiento D = Cs - Cr.

    El mapa de calor muestra:
        - Incrementos positivos/negativos en la correlación entre articulaciones.
        - Una leyenda con el valor medio absoluto de la diferencia (mean|Δ|).
    """
    D = Cs - Cr
    J = D.shape[0]

    mean_abs = float(np.mean(np.abs(D)))

    plt.figure(figsize=(4.2 + 0.25 * J, 3.8 + 0.25 * J), dpi=140)
    im = plt.imshow(D, vmin=-1, vmax=1, cmap='bwr')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = _joint_labels(J)
    plt.xticks(range(J), ticks, rotation=45, ha='right')
    plt.yticks(range(J), ticks)
    _annotate_cells(plt.gca(), D, annotate_fmt)

    plt.title(
        'Matriz de acoplamiento: Δ (Sintéticos - Reales)\n'
        f'mean|Δ|={mean_abs:.4f}'
    )

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def _draw_cov_ellipse(ax, mean, cov, nsig=1.0, lw=2.0, ls='-',
                      edgecolor='#003366'):
    """
    Dibuja sobre el eje ax una elipse de covarianza centrada en `mean`
    con matriz de covarianza `cov`, correspondiente a nsig desviaciones
    estándar.

    Este tipo de elipses permite visualizar la dispersión principal de
    la nube de puntos en el espacio bidimensional (PC1, PC2).
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nsig * np.sqrt(np.clip(vals, 1e-9, None))

    from matplotlib.patches import Ellipse
    e = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=theta,
        fill=False,
        lw=lw,
        ls=ls,
        edgecolor=edgecolor
    )
    ax.add_patch(e)


def save_pca_ellipses(Zr, Zs, var_sum, out_ell):
    """
    Representa el solapamiento entre las muestras reales y sintéticas en
    el plano PC1–PC2, incluyendo:

        - Nubes de puntos para datos reales y sintéticos.
        - Centroides de cada nube.
        - Elipses de 1σ de dispersión alrededor de cada centroide.
        - Varianza explicada total por PC1+PC2 en el título.
    """
    # Colores predefinidos para cada conjunto
    real_color_pts = (0.2, 0.45, 0.8, 0.35)
    synth_color_pts = (0.98, 0.5, 0.1, 0.35)
    real_centroid = '#003366'
    synth_centroid = '#AA0000'

    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(Zr[:, 0], Zr[:, 1], s=10, alpha=0.5,
                label='Real', color=real_color_pts)
    plt.scatter(Zs[:, 0], Zs[:, 1], s=10, alpha=0.5,
                label='Synth', marker='x', color=synth_color_pts)

    # Cálculo de centroides y covarianzas
    cr, cs = Zr.mean(0), Zs.mean(0)
    cov_r = np.cov(Zr.T)
    cov_s = np.cov(Zs.T)

    plt.scatter([cr[0]], [cr[1]], marker='D', s=70,
                label='Centroide Real', color=real_centroid)
    plt.scatter([cs[0]], [cs[1]], marker='D', s=70,
                label='Centroide Synth', color=synth_centroid)

    ax = plt.gca()
    _draw_cov_ellipse(ax, cr, cov_r, nsig=1.0, lw=2.0, ls='-',
                      edgecolor=real_centroid)
    _draw_cov_ellipse(ax, cs, cov_s, nsig=1.0, lw=2.0, ls='--',
                      edgecolor=synth_centroid)

    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.title(f'PCA (var_expl={var_sum:.2f}) — centroides + elipses 1σ')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(out_ell)
    plt.close()


def save_curves_mean_sd(Ym_r, Sd_r, Ym_s, Sd_s, rmse_mu, r_mu, outpath):
    """
    Dibuja, por articulación, las curvas promedio reales y sintéticas con sus
    bandas de ±1σ, y anota en cada subplot las métricas:

        - RMSE(μ–μ) en grados.
        - r(μ–μ), coeficiente de correlación de Pearson de las curvas medias.

    La figura resultante resume la concordancia temporal entre las trayectorias
    medias reales y generadas para cada articulación.
    """
    T, J = Ym_r.shape
    t = np.linspace(0, 100, T, dtype=np.float32)  # porcentaje de ciclo
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    axes = axes.ravel()
    labels = _joint_labels(J)

    for j in range(J):
        ax = axes[j]

        # Curvas promedio reales y sintéticas
        ax.plot(t, Ym_r[:, j], label='Real μ', color='#1f77b4')
        ax.plot(t, Ym_s[:, j], '--', label='Synth μ', color='#ff7f0e')

        # Bandas de ±1σ
        ax.fill_between(t, Ym_r[:, j] - Sd_r[:, j], Ym_r[:, j] + Sd_r[:, j],
                        alpha=0.25, label='Real ±1σ', color='#1f77b4')
        ax.fill_between(t, Ym_s[:, j] - Sd_s[:, j], Ym_s[:, j] + Sd_s[:, j],
                        alpha=0.20, label='Synth ±1σ', color='#ff7f0e')

        ax.set_title(labels[j])
        ax.set_xlabel('% ciclo')
        ax.set_ylabel('ángulo (°)')
        ax.grid(alpha=0.2)

        # Anotación de métricas locales
        txt = f"RMSE: {rmse_mu[j]:.3f}°\n r: {r_mu[j]:.3f}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white',
                          alpha=0.85, edgecolor='0.3'))

    # Leyenda común para todas las articulaciones
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc='upper center', ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(outpath)
    plt.close()


# ------------------------------------------------------------
# Pipeline de evaluación y generación de informe
# ------------------------------------------------------------
def run_compare(real_mat, synth_mat, outdir):
    """
    Ejecuta la comparación completa entre un conjunto real y uno sintético:

        1) Carga y normaliza las dimensiones de ambos archivos .mat.
        2) Calcula curvas promedio y desviaciones estándar.
        3) Evalúa métricas μ–μ (RMSE, r) por articulación y %RMSE
           relativo a la amplitud de la curva media real.
        4) Construye matrices de acoplamiento interarticular y su diferencia.
        5) Realiza PCA conjunto y representa el solapamiento con elipses.
        6) Genera figuras asociadas y un informe de texto con todas las métricas.
    """
    os.makedirs(outdir, exist_ok=True)

    # Carga de datos real y sintético
    Yr, Xr, Sr = load(real_mat)
    Ys, Xs, Ss = load(synth_mat)

    # Curvas media y desviación estándar
    Ym_r, Sd_r = curve_mean_sd(Yr)
    Ym_s, Sd_s = curve_mean_sd(Ys)

    # Métricas μ–μ por articulación
    rmse_mu = rmse_avgcurve(Ym_r, Ym_s)
    r_mu = pearson_r_avgcurve(Ym_r, Ym_s)

    # Amplitud fisiológica aproximada y RMSE relativo
    amp_r = joint_amp_from_mean(Ym_r)
    rmse_rel = 100.0 * rmse_mu / (amp_r + 1e-9)

    # Coupling interarticular
    C_r = coupling_matrix(Yr)
    C_s = coupling_matrix(Ys)

    save_coupling_heatmap(
        C_r,
        os.path.join(outdir, 'coupling_real.png'),
        'Matriz de acoplamiento: Datos Reales',
        vmin=-1, vmax=1,
        annotate_fmt='{:.2f}'
    )
    save_coupling_heatmap(
        C_s,
        os.path.join(outdir, 'coupling_synth.png'),
        'Matriz de acoplamiento: Datos Sintéticos',
        vmin=-1, vmax=1,
        annotate_fmt='{:.2f}'
    )
    save_coupling_delta(
        C_r, C_s,
        os.path.join(outdir, 'coupling_delta.png'),
        annotate_fmt='{:+.2f}'
    )

    dC_meanabs = coupling_delta_scalar(C_r, C_s, mode="mean_abs")
    dC_fro = coupling_delta_scalar(C_r, C_s, mode="fro_norm")

    # PCA sobre Yr + Ys
    Zr, Zs, var_sum, cent = pca_overlap_svd(Yr, Ys)

    # Filtrado de outliers extremos para la visualización (no afecta a var_sum ni cent)
    mask_r = (np.abs(Zr[:, 0]) < 100) & (np.abs(Zr[:, 1]) < 100)
    mask_s = (np.abs(Zs[:, 0]) < 100) & (np.abs(Zs[:, 1]) < 100)
    Zr, Zs = Zr[mask_r], Zs[mask_s]

    save_pca_ellipses(
        Zr, Zs, var_sum,
        os.path.join(outdir, 'pca_compare_ellipses.png')
    )

    # Curvas promedio ±1σ con métricas μ-μ anotadas
    save_curves_mean_sd(
        Ym_r, Sd_r, Ym_s, Sd_s, rmse_mu, r_mu,
        os.path.join(outdir, 'curves_mean_sd.png')
    )

    # --------------------------------------------------------
    # Informe de texto con resumen de resultados
    # --------------------------------------------------------
    names = _joint_labels(Yr.shape[2])
    rep = os.path.join(outdir, 'report.txt')

    with open(rep, 'w', encoding='utf-8') as f:
        f.write('# COMPARE: Real vs Synth (compacto)\n')
        f.write(
            f'REAL : {os.path.basename(real_mat)}  '
            f'N={Yr.shape[0]} T={Yr.shape[1]} J={Yr.shape[2]}\n'
        )
        f.write(
            f'SYNTH: {os.path.basename(synth_mat)} '
            f'N={Ys.shape[0]} T={Ys.shape[1]} J={Ys.shape[2]}\n\n'
        )

        f.write('[A] Curvas promedio (μ ±1σ) con RMSE y r anotados por panel\n')
        f.write('Figura: curves_mean_sd.png\n\n')

        f.write('[B] Métricas μ–μ por articulación (para referencia)\n')
        f.write('Ángulo              | rango(°) |  RMSEμ-μ |   r(μ-μ) |  %RMSE (RMSE/rango)\n')
        f.write('-' * 88 + '\n')
        for j, name in enumerate(names):
            f.write(
                f'{name:<18} | {amp_r[j]:8.3f} | {rmse_mu[j]:8.3f} | '
                f'{r_mu[j]:7.3f} | {rmse_rel[j]:8.2f}%\n'
            )

        f.write('\n[C] Coupling interarticular (con valores en figuras)\n')
        f.write(
            'Figuras : coupling_real.png, coupling_synth.png, '
            'coupling_delta.png\n'
        )
        f.write(
            f'Resumen Δ: mean|Δ|={dC_meanabs:.4f}  '
            f'Fro(norm)={dC_fro:.4f}\n'
        )

        f.write('\n[D] PCA (elipses 1σ + centroides)\n')
        f.write(f'Varianza explicada PC1+PC2: {var_sum:.3f}\n')
        f.write(f'Distancia de centroides (PC1-2): {cent:.3f}\n')
        f.write('Figura: pca_compare_ellipses.png\n')

        f.write('\nFin.\n')

    print(f'>> Informe: {rep}')


# ------------------------------------------------------------
# Interfaz de línea de comandos
# ------------------------------------------------------------
def main():
    """
    Punto de entrada de la línea de comandos.

    Argumentos
    ----------
    --real_mat  : ruta al archivo .mat con datos reales.
    --synth_mat : ruta al archivo .mat con datos sintéticos.
    --out       : carpeta de salida para figuras e informe.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--real_mat', type=str, required=True)
    ap.add_argument('--synth_mat', type=str, required=True)
    ap.add_argument('--out', type=str, default='compare_report')
    args = ap.parse_args()

    run_compare(args.real_mat, args.synth_mat, args.out)


if __name__ == '__main__':
    main()
