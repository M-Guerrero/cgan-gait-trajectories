import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# UTILIDADES PARA EL ESTUDIO DE L,H (longitud y altura de paso)
# ============================================================

def ensure_N2(X, name="X"):
    """
    Garantiza que la matriz X tenga forma (N, 2), donde N es el número
    de muestras y las dos columnas corresponden a [L, H].

    Admite dos formatos de entrada:
        - (N, 2): cada fila es una muestra con [L, H].
        - (2, N): dos filas (L y H) y N columnas.

    En ambos casos, devuelve una matriz (N, 2).

    Parámetros
    ----------
    X : array-like
        Matriz con dimensiones (N, 2) o (2, N).
    name : str
        Nombre identificativo de la variable (para mensajes de error).

    Devuelve
    --------
    X_out : np.ndarray
        Matriz con forma (N, 2).

    Lanza
    -----
    ValueError
        Si X no es 2D o no coincide con las formas esperadas.
    """
    X = np.array(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"{name} debe ser 2D, pero tiene dims={X.ndim} y shape={X.shape}")

    if X.shape[1] == 2:
        return X
    if X.shape[0] == 2:
        return X.T

    raise ValueError(
        f"{name} tiene shape {X.shape}, no coincide con (N, 2) ni con (2, N)."
    )


def load_LH_from_mat(path, printer=print):
    """
    Carga las variables X_real y X_cond desde un archivo .mat generado
    con VALID_CLINICA y aplica un filtrado de calidad sobre L y H.

    El procedimiento realiza:
        1) Normalización de X_real y X_cond a forma (N, 2).
        2) Filtro de muestras con valores NaN o Inf en cualquiera de las
           dos matrices.
        3) Filtro de valores no físicos (L <= 0 o H <= 0) tanto en
           X_cond como en X_real.
        4) Construcción de una máscara global de muestras válidas.

    Parámetros
    ----------
    path : str
        Ruta al archivo .mat generado por VALID_CLINICA.
    printer : callable
        Función de impresión utilizada para el log (por defecto, print).

    Devuelve
    --------
    X_real_clean : np.ndarray
        Matriz (N_val, 2) con valores reconstruidos por el modelo cinemático.
    X_cond_clean : np.ndarray
        Matriz (N_val, 2) con las condiciones de entrada utilizadas.

    Lanza
    -----
    ValueError
        Si faltan variables requeridas o si todas las muestras son descartadas.
    """
    with h5py.File(path, "r") as f:
        if "X_real" not in f:
            raise ValueError(
                "El archivo no contiene 'X_real'. "
                "Es necesario utilizar la versión actualizada de VALID_CLINICA."
            )
        if "X_cond" not in f:
            raise ValueError(
                "El archivo no contiene 'X_cond'. "
                "Este script está diseñado para datasets sintéticos condicionados."
            )

        X_real = np.array(f["X_real"])
        X_cond = np.array(f["X_cond"])

    # Normalización de dimensiones a (N, 2)
    X_real = ensure_N2(X_real, "X_real")
    X_cond = ensure_N2(X_cond, "X_cond")

    if X_real.shape[0] != X_cond.shape[0]:
        raise ValueError(
            f"N no coincide entre X_real y X_cond: "
            f"X_real={X_real.shape}, X_cond={X_cond.shape}"
        )

    N_total = X_real.shape[0]

    # 1) Filtro de NaNs/Inf en ambas matrices
    finite_real = np.all(np.isfinite(X_real), axis=1)
    finite_cond = np.all(np.isfinite(X_cond), axis=1)
    mask_finite = finite_real & finite_cond
    n_finite = int(mask_finite.sum())
    n_nan_disc = N_total - n_finite

    if n_nan_disc > 0:
        printer(
            f"[WARN] Se han descartado {n_nan_disc} de {N_total} muestras "
            f"por contener NaNs/Inf en X_cond o X_real "
            f"({100.0 * n_nan_disc / N_total:.1f}%)."
        )
    else:
        printer(f"[OK] {N_total}/{N_total} muestras sin NaNs/Inf en X_cond/X_real.")

    # 2) Filtro de valores no físicos: L <= 0 o H <= 0
    cond_nonpos = (X_cond[:, 0] <= 0) | (X_cond[:, 1] <= 0)
    real_nonpos = (X_real[:, 0] <= 0) | (X_real[:, 1] <= 0)

    bad_cond_only = cond_nonpos & ~real_nonpos
    bad_real_only = real_nonpos & ~cond_nonpos
    bad_both = cond_nonpos & real_nonpos

    n_cond_only = int(bad_cond_only.sum())
    n_real_only = int(bad_real_only.sum())
    n_both = int(bad_both.sum())

    disc_nonpos_mask = bad_cond_only | bad_real_only | bad_both
    n_disc_nonpos_total = int(disc_nonpos_mask.sum())

    if n_disc_nonpos_total > 0:
        printer("\n[WARN] Filtro de L/H no físicos (<= 0 en alguna componente):")
        if n_cond_only > 0:
            printer(
                f"  - Solo por X_cond <= 0 : {n_cond_only} "
                f"({100.0 * n_cond_only / N_total:.1f}%)"
            )
        if n_real_only > 0:
            printer(
                f"  - Solo por X_real <= 0 : {n_real_only} "
                f"({100.0 * n_real_only / N_total:.1f}%)"
            )
        if n_both > 0:
            printer(
                f"  - Por ambos (cond y real <= 0): {n_both} "
                f"({100.0 * n_both / N_total:.1f}%)"
            )
        printer(
            f"  - TOTAL descartadas por L/H <= 0: {n_disc_nonpos_total} "
            f"({100.0 * n_disc_nonpos_total / N_total:.1f}%)\n"
        )
    else:
        printer("[OK] Ninguna muestra con L/H <= 0 en cond o real.")

    # 3) Máscara global de muestras válidas
    mask_valid = mask_finite & ~disc_nonpos_mask
    n_valid = int(mask_valid.sum())

    if n_valid == 0:
        raise ValueError(
            "Todas las muestras han sido descartadas por NaNs/Inf o L/H <= 0."
        )

    if n_valid < N_total:
        printer(
            f"[INFO] En total se han descartado {N_total - n_valid} de {N_total} "
            f"muestras ({100.0 * (N_total - n_valid) / N_total:.1f}%) "
            f"por filtros de calidad."
        )
        printer(
            f"[OK] Quedan {n_valid} muestras válidas "
            f"({100.0 * n_valid / N_total:.1f}%)."
        )
    else:
        printer(f"[OK] Todas las {N_total} muestras pasan los filtros de calidad.")

    X_real = X_real[mask_valid]
    X_cond = X_cond[mask_valid]

    return X_real, X_cond


def plot_LH_plane(X_cond, X_real, out_png="LH_plane.png", title_suffix=""):
    """
    Representa el plano (L, H) comparando las condiciones de entrada
    con los valores reconstruidos por el modelo cinemático.

    Parámetros
    ----------
    X_cond : np.ndarray
        Matriz (N, 2) con [L_cond, H_cond].
    X_real : np.ndarray
        Matriz (N, 2) con [L_real, H_real].
    out_png : str
        Ruta de salida para la figura.
    title_suffix : str
        Sufijo opcional para el título (indicador de sesión o dataset).
    """
    Lc, Hc = X_cond[:, 0], X_cond[:, 1]
    Lr, Hr = X_real[:, 0], X_real[:, 1]

    plt.figure(figsize=(8, 6), dpi=140)
    plt.scatter(Lc, Hc, s=25, alpha=0.4,
                label="Condiciones de entrada (X_cond)")
    plt.scatter(Lr, Hr, s=25, alpha=0.4,
                label="Tras cinemática (X_real)")

    plt.xlabel("Longitud de paso L (m)")
    plt.ylabel("Altura de paso H (m)")
    plt.title(f"Plano L–H: condiciones vs cinemática {title_suffix}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[FIG] Guardado plano L–H en: {out_png}")


def plot_LH_identity(X_cond, X_real, out_png="LH_identity.png", title_suffix=""):
    """
    Representa dos diagramas de dispersión:

        - L_cond (eje x) frente a L_real (eje y).
        - H_cond (eje x) frente a H_real (eje y).

    En cada panel se incluiye la recta de identidad y = x para facilitar
    la inspección del ajuste entre condición impuesta y valor reconstruido.

    Parámetros
    ----------
    X_cond : np.ndarray
        Matriz (N, 2) con [L_cond, H_cond].
    X_real : np.ndarray
        Matriz (N, 2) con [L_real, H_real].
    out_png : str
        Ruta de salida para la figura.
    title_suffix : str
        Sufijo opcional de contexto para el título general.
    """
    Lc, Hc = X_cond[:, 0], X_cond[:, 1]
    Lr, Hr = X_real[:, 0], X_real[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140, sharex=False, sharey=False)
    fig.suptitle(f"Comparación condiciones vs cinemática {title_suffix}", fontsize=12)

    # Panel para L
    ax = axes[0]
    ax.scatter(Lc, Lr, s=20, alpha=0.4)
    ax.set_xlabel("L_cond (condición de entrada) [m]")
    ax.set_ylabel("L_real (modelo cinemático) [m]")
    ax.set_title("Longitud de paso")

    L_min_common = min(Lc.min(), Lr.min())
    L_max_common = max(Lc.max(), Lr.max())
    xL = np.linspace(L_min_common, L_max_common, 100)
    ax.plot(xL, xL, linewidth=2, linestyle="--")

    def _margin(min_v, max_v):
        span = max_v - min_v
        return 0.05 * span if span > 0 else 0.05

    x_margin_L = _margin(Lc.min(), Lc.max())
    y_margin_L = _margin(Lr.min(), Lr.max())
    ax.set_xlim(Lc.min() - x_margin_L, Lc.max() + x_margin_L)
    ax.set_ylim(Lr.min() - y_margin_L, Lr.max() + y_margin_L)
    ax.grid(alpha=0.25)

    # Panel para H
    ax = axes[1]
    ax.scatter(Hc, Hr, s=20, alpha=0.4)
    ax.set_xlabel("H_cond (condición de entrada) [m]")
    ax.set_ylabel("H_real (modelo cinemático) [m]")
    ax.set_title("Altura de paso")

    H_min_common = min(Hc.min(), Hr.min())
    H_max_common = max(Hc.max(), Hr.max())
    xH = np.linspace(H_min_common, H_max_common, 100)
    ax.plot(xH, xH, linewidth=2, linestyle="--")

    x_margin_H = _margin(Hc.min(), Hc.max())
    y_margin_H = _margin(Hr.min(), Hr.max())
    ax.set_xlim(Hc.min() - x_margin_H, Hc.max() + x_margin_H)
    ax.set_ylim(Hr.min() - y_margin_H, Hr.max() + y_margin_H)
    ax.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_png)
    plt.close(fig)
    print(f"[FIG] Guardado L_cond/H_cond vs L_real/H_real en: {out_png}")


# ------------------------------------------------------------
# Detección y filtrado de outliers de L,H mediante Tukey
# ------------------------------------------------------------

def count_outliers_Tukey(arr):
    """
    Identifica outliers en un vector unidimensional utilizando el criterio
    clásico de Tukey (basado en el rango intercuartílico).

        - Q1: percentil 25
        - Q3: percentil 75
        - IQR = Q3 - Q1
        - Intervalo aceptable: [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

    Parámetros
    ----------
    arr : array-like
        Vector de datos.

    Devuelve
    --------
    mask_out : np.ndarray (bool)
        Máscara booleana que indica las muestras marcadas como outlier.
    n_out : int
        Número total de outliers.
    lower : float
        Límite inferior del intervalo aceptable.
    upper : float
        Límite superior del intervalo aceptable.
    """
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask_out = (arr < lower) | (arr > upper)
    n_out = int(mask_out.sum())
    return mask_out, n_out, lower, upper


def filter_outliers_LH(X_cond, X_real, printer=print):
    """
    Aplica un filtro de outliers a las trayectorias L_real y H_real
    utilizando el criterio de Tukey y descarta aquellas muestras que
    se consideren atípicas en L o en H.

    Parámetros
    ----------
    X_cond : np.ndarray
        Matriz (N, 2) con [L_cond, H_cond].
    X_real : np.ndarray
        Matriz (N, 2) con [L_real, H_real].
    printer : callable
        Función de impresión utilizada para el log.

    Devuelve
    --------
    X_cond_filt : np.ndarray
        Matriz (N_filtrado, 2) con condiciones tras eliminar outliers.
    X_real_filt : np.ndarray
        Matriz (N_filtrado, 2) con valores reales tras eliminar outliers.
    mask_keep : np.ndarray (bool)
        Máscara booleana de muestras conservadas.
    """
    Lr, Hr = X_real[:, 0], X_real[:, 1]

    mask_L_out, n_out_L, lowL, upL = count_outliers_Tukey(Lr)
    mask_H_out, n_out_H, lowH, upH = count_outliers_Tukey(Hr)

    mask_out_any = mask_L_out | mask_H_out
    mask_keep = ~mask_out_any

    n_total = len(Lr)
    n_disc = int(mask_out_any.sum())

    printer("\n[INFO] Filtro adicional de outliers (Tukey) sobre X_real:")
    printer(f"   - Outliers en L_real: {n_out_L} (umbral [{lowL:.3f}, {upL:.3f}])")
    printer(f"   - Outliers en H_real: {n_out_H} (umbral [{lowH:.3f}, {upH:.3f}])")
    printer(
        f"   - Muestras descartadas por ser outlier en L o H: "
        f"{n_disc} de {n_total} "
        f"({100.0 * n_disc / n_total:.1f}%)"
    )
    printer(
        f"   - Muestras restantes para estadísticas L/H: "
        f"{n_total - n_disc} ({100.0 * (n_total - n_disc) / n_total:.1f}%)\n"
    )

    X_real_filt = X_real[mask_keep, :]
    X_cond_filt = X_cond[mask_keep, :]

    return X_cond_filt, X_real_filt, mask_keep


def compute_stats(X_cond, X_real, printer=print):
    """
    Calcula estadísticas de concordancia entre las condiciones de entrada
    (X_cond) y los valores reconstruidos por el modelo cinemático (X_real)
    para la longitud y altura de paso.

    Métricas calculadas:
        - MAE (error absoluto medio).
        - STD del error absoluto.
        - RMSE.
        - Correlación de Pearson.
        - Percentiles del error absoluto (p50, p75, p95).

    Parámetros
    ----------
    X_cond : np.ndarray
        Matriz (N, 2) con [L_cond, H_cond].
    X_real : np.ndarray
        Matriz (N, 2) con [L_real, H_real].
    printer : callable
        Función de impresión para registrar resultados.

    Devuelve
    --------
    dL : np.ndarray
        Diferencias L_real - L_cond por muestra.
    dH : np.ndarray
        Diferencias H_real - H_cond por muestra.
    """
    Lc, Hc = X_cond[:, 0], X_cond[:, 1]
    Lr, Hr = X_real[:, 0], X_real[:, 1]

    dL = Lr - Lc
    dH = Hr - Hc

    abs_dL = np.abs(dL)
    abs_dH = np.abs(dH)

    def safe_corr(a, b):
        if np.all(a == a[0]) or np.all(b == b[0]):
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    printer("\n========== ESTADÍSTICAS L y H ==========\n")

    printer(">> L (longitud de paso):")
    printer(f"  N = {len(Lc)} muestras válidas (después de filtros y outliers)")
    printer(f"  MAE(|dL|)              = {np.mean(abs_dL):.4f} m")
    printer(f"  std(|dL|)              = {np.std(abs_dL):.4f} m")
    printer(f"  RMSE(L_real, L_cond)   = {np.sqrt(np.mean(dL**2)):.4f} m")
    printer(f"  corr(L_real, L_cond)   = {safe_corr(Lr, Lc):.4f}")
    printer(
        f"  percentiles |dL| (m):  "
        f"p50 = {np.percentile(abs_dL, 50):.4f}, "
        f"p75 = {np.percentile(abs_dL, 75):.4f}, "
        f"p95 = {np.percentile(abs_dL, 95):.4f}"
    )

    printer("\n>> H (altura de paso):")
    printer(f"  MAE(|dH|)              = {np.mean(abs_dH):.4f} m")
    printer(f"  std(|dH|)              = {np.std(abs_dH):.4f} m")
    printer(f"  RMSE(H_real, H_cond)   = {np.sqrt(np.mean(dH**2)):.4f} m")
    printer(f"  corr(H_real, H_cond)   = {safe_corr(Hr, Hc):.4f}")
    printer(
        f"  percentiles |dH| (m):  "
        f"p50 = {np.percentile(abs_dH, 50):.4f}, "
        f"p75 = {np.percentile(abs_dH, 75):.4f}, "
        f"p95 = {np.percentile(abs_dH, 95):.4f}"
    )

    printer("\n========================================\n")

    return dL, dH


# ============================================================
# ANÁLISIS DE ERRORES EN FUNCIÓN DE LA CONDICIÓN (QUESITOS + BOXPLOTS)
# ============================================================

def _make_binned_error_boxplot(ax, cond_values, errors_cm, n_bins, titulo, ylabel):
    """
    Construye un diagrama de caja (boxplot) del error (en cm) agrupado
    por intervalos de la condición (L_cond o H_cond).

    La partición en tramos se realiza mediante cuantiles, generando
    aproximadamente el mismo número de muestras por bin.

    Parámetros
    ----------
    ax : matplotlib.axes.Axes
        Eje sobre el que se dibuja el boxplot.
    cond_values : np.ndarray
        Valores de condición (L_cond o H_cond) en metros.
    errors_cm : np.ndarray
        Errores firmados correspondientes en centímetros.
    n_bins : int
        Número de intervalos (bins) definidos por cuantiles.
    titulo : str
        Título del panel.
    ylabel : str
        Etiqueta del eje y.

    Devuelve
    --------
    edges : np.ndarray
        Valores de los bordes de los bins (en metros).
    labels : list
        Etiquetas de texto para cada bin (en cm).
    data_box : list of np.ndarray
        Lista de arrays con los errores en cada bin.
    """
    cond_values = np.asarray(cond_values, dtype=float)
    errors_cm = np.asarray(errors_cm, dtype=float)

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(cond_values, qs)

    data_box = []
    labels = []

    for i in range(n_bins):
        low = edges[i]
        high = edges[i + 1]

        if i < n_bins - 1:
            mask = (cond_values >= low) & (cond_values < high)
        else:
            mask = (cond_values >= low) & (cond_values <= high)

        errs_bin = errors_cm[mask]
        if errs_bin.size == 0:
            continue

        data_box.append(errs_bin)
        low_cm = low * 100.0
        high_cm = high * 100.0
        labels.append(f"{low_cm:.0f}–{high_cm:.0f} cm")

    ax.boxplot(
        data_box,
        labels=labels,
        showmeans=False,
        flierprops=dict(
            marker="o",
            markeredgecolor="red",
            markersize=4,
            linestyle="none",
        ),
        boxprops=dict(color="navy"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="navy"),
        capprops=dict(color="navy"),
    )
    ax.set_title(titulo)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    return edges, labels, data_box


def plot_error_and_value_distributions(X_cond, X_real, dL, dH,
                                       out_dir=".", prefix="", printer=print,
                                       n_bins=3):
    """
    Genera un resumen gráfico del error de L y H que incluye:

        1) Una figura con dos gráficos de sectores ("quesitos"):
           - Distribución de |dL| en cm según rangos predefinidos.
           - Distribución de |dH| en cm según rangos predefinidos.

        2) Una figura con diagramas de caja del error firmado en cm
           (dL, dH) estratificado en función de la condición (L_cond, H_cond),
           utilizando bins definidos por cuantiles.

    Parámetros
    ----------
    X_cond : np.ndarray
        Matriz (N, 2) con [L_cond, H_cond] en metros.
    X_real : np.ndarray
        Matriz (N, 2) con [L_real, H_real] en metros.
    dL : np.ndarray
        Diferencias L_real - L_cond en metros.
    dH : np.ndarray
        Diferencias H_real - H_cond en metros.
    out_dir : str
        Carpeta de salida para las figuras.
    prefix : str
        Prefijo opcional para los nombres de archivo de las figuras.
    printer : callable
        Función de impresión utilizada para registrar el resumen.
    n_bins : int
        Número de tramos de condición para los boxplots.
    """
    suf = f"({prefix})" if prefix else ""

    # ----- QUESITOS: distribución global de |error| -----
    absL_cm = np.abs(dL) * 100.0
    absH_cm = np.abs(dH) * 100.0

    # Rangos de error (en cm) para longitud y altura
    bins_L_cm = [0.0, 2.0, 5.0, 10.0, np.inf]
    labels_L = ["0–2 cm", "2–5 cm", "5–10 cm", "≥10 cm"]

    bins_H_cm = [0.0, 1.0, 2.0, 5.0, np.inf]
    labels_H = ["0–1 cm", "1–2 cm", "2–5 cm", "≥5 cm"]

    counts_L, _ = np.histogram(absL_cm, bins=bins_L_cm)
    counts_H, _ = np.histogram(absH_cm, bins=bins_H_cm)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=140)
    fig.suptitle(f"Distribución de errores |dL| y |dH| {suf}", fontsize=13)

    def autopct_fmt(pct):
        return f"{pct:.1f}%"

    axes[0].pie(counts_L, labels=labels_L, autopct=autopct_fmt)
    axes[0].set_title("|dL|")

    axes[1].pie(counts_H, labels=labels_H, autopct=autopct_fmt)
    axes[1].set_title("|dH|")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    pie_path = os.path.join(out_dir, f"{prefix}LH_error_pies.png")
    plt.savefig(pie_path)
    plt.close(fig)
    print(f"[FIG] Guardado quesitos conjuntos de error en: {pie_path}")

    # ----- BOXPLOTS: error por tramos de la condición X_cond -----
    Lc = X_cond[:, 0]
    Hc = X_cond[:, 1]

    dL_cm = dL * 100.0
    dH_cm = dH * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=140)
    fig.suptitle(f"Error de L_cond y H_cond {suf}", fontsize=13)

    # Error en L por tramos de L_cond
    edges_L, labels_Lbins, data_Lbins = _make_binned_error_boxplot(
        axes[0],
        Lc,
        dL_cm,
        n_bins=n_bins,
        titulo="Error L_real - L_cond por tramos de L_cond",
        ylabel="Error L_real - L_cond (cm)",
    )

    # Error en H por tramos de H_cond
    edges_H, labels_Hbins, data_Hbins = _make_binned_error_boxplot(
        axes[1],
        Hc,
        dH_cm,
        n_bins=n_bins,
        titulo="Error H_real - H_cond por tramos de H_cond",
        ylabel="Error H_real - H_cond (cm)",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    box_path = os.path.join(out_dir, f"{prefix}LH_error_boxplots_condbins.png")
    plt.savefig(box_path)
    plt.close(fig)
    print(f"[FIG] Guardado boxplots de error de condición en: {box_path}")

    # Resumen numérico por bin: media (sesgo) y desviación típica del error firmado
    printer("\n[INFO] Error en L de L_cond (en cm, firmado):")
    for label, errs in zip(labels_Lbins, data_Lbins):
        mean_bin = float(np.mean(errs))
        std_bin = float(np.std(errs))
        n_bin = errs.size
        printer(
            f"   Bin {label}: N={n_bin}, media(error) = {mean_bin:.2f} cm, "
            f"std = {std_bin:.2f} cm"
        )

    printer("\n[INFO] Error en H de H_cond (en cm, firmado):")
    for label, errs in zip(labels_Hbins, data_Hbins):
        mean_bin = float(np.mean(errs))
        std_bin = float(np.std(errs))
        n_bin = errs.size
        printer(
            f"   Bin {label}: N={n_bin}, media(error) = {mean_bin:.2f} cm, "
            f"std = {std_bin:.2f} cm"
        )

    printer(f"\n[OK] Quesitos y boxplots de error de condición guardados en {out_dir}\n")


# ============================================================
# UTILIDADES PARA EL ANÁLISIS DE TRAYECTORIAS DEL PIE (Y_pie)
# ============================================================

def load_feet_from_mat(path):
    """
    Carga la variable Y_pie desde el archivo .mat y la reorganiza
    a formato (N, T, 8), donde las 8 columnas corresponden a:

        [R_ankle_X, R_ankle_Z, R_heel_Z, R_toe_Z,
         L_ankle_X, L_ankle_Z, L_heel_Z, L_toe_Z]

    Se soportan distintas disposiciones típicas:
        - (N, T, 8)
        - (8, T, N)
        - (T, 8, N)

    Parámetros
    ----------
    path : str
        Ruta al archivo .mat producido por VALID_CLINICA.

    Devuelve
    --------
    Y_pie_ntk : np.ndarray
        Tensor con forma (N, T, 8).
    var_names : list of str
        Nombres de las variables en el mismo orden que el eje 2 de Y_pie_ntk.

    Lanza
    -----
    ValueError
        Si no se encuentra la variable Y_pie o la forma no coincide con
        ningún caso soportado.
    """
    with h5py.File(path, "r") as f:
        if "Y_pie" not in f:
            raise ValueError("El archivo no contiene 'Y_pie' (trayectorias del pie).")
        Y_pie = np.array(f["Y_pie"], dtype=np.float32)

    if Y_pie.ndim != 3:
        raise ValueError(f"Y_pie debe ser 3D, pero tiene shape {Y_pie.shape}")

    s0, s1, s2 = Y_pie.shape

    # Reorganización a formato (N, T, 8)
    if s2 == 8:
        Y_pie_ntk = Y_pie
    elif s0 == 8:
        Y_pie_ntk = np.transpose(Y_pie, (2, 1, 0))
    elif s1 == 8:
        Y_pie_ntk = np.transpose(Y_pie, (2, 0, 1))
    else:
        raise ValueError(
            f"Y_pie tiene shape {Y_pie.shape}, no cuadra con ninguna "
            "de las formas esperadas ((N,T,8), (8,T,N), (T,8,N))."
        )

    var_names = [
        "R_ankle_X", "R_ankle_Z", "R_heel_Z", "R_toe_Z",
        "L_ankle_X", "L_ankle_Z", "L_heel_Z", "L_toe_Z"
    ]
    return Y_pie_ntk, var_names


def _plot_foot_channel_on_ax(ax, t, traj, title):
    """
    Representa múltiples trayectorias de un mismo canal de pie como
    un conjunto de curvas semitransparentes, junto con la media y
    la banda de ±1 desviación estándar.

    Parámetros
    ----------
    ax : matplotlib.axes.Axes
        Eje sobre el que se dibujan las curvas.
    t : np.ndarray
        Vector temporal (en % del ciclo).
    traj : np.ndarray
        Tensor (N, T) con las trayectorias del canal seleccionado.
    title : str
        Título del subplot.
    """
    for i in range(traj.shape[0]):
        ax.plot(t, traj[i, :], alpha=0.02, linewidth=0.5)

    mean_traj = np.nanmean(traj, axis=0)
    std_traj = np.nanstd(traj, axis=0)

    ax.plot(t, mean_traj, linewidth=2.0, label="Media")
    ax.fill_between(t, mean_traj - std_traj, mean_traj + std_traj,
                    alpha=0.3, label="Media ± 1·std")

    ax.set_title(title)
    ax.set_xlabel("% del ciclo de la marcha")
    ax.set_ylabel("Posición / altura (m)")
    ax.grid(alpha=0.25)


def plot_feet_X(Y_pie, var_names, out_png):
    """
    Genera una figura con las trayectorias de X (avance) del tobillo
    derecho e izquierdo, mostrando la dispersión inter-ciclo y la
    envolvente media ±1 desviación estándar.

    Parámetros
    ----------
    Y_pie : np.ndarray
        Tensor (N, T, 8) con las trayectorias de marcadores del pie.
    var_names : list of str
        Nombres asociados a las columnas de Y_pie.
    out_png : str
        Ruta de salida para la figura.
    """
    N, T, K = Y_pie.shape
    t = np.linspace(0, 100, T)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140, sharex=True)
    fig.suptitle("Trayectorias del pie en X", fontsize=14)

    order_names = ["R_ankle_X", "L_ankle_X"]
    axes = axes.ravel()
    for ax, name in zip(axes, order_names):
        idx = var_names.index(name)
        _plot_foot_channel_on_ax(ax, t, Y_pie[:, :, idx], name)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[FIG] Guardado foot X en: {out_png}")


def plot_feet_Z(Y_pie, var_names, out_png):
    """
    Genera una figura con las trayectorias verticales (Z) de tobillo,
    talón y antepié (heel/toe) de ambos pies, incluyendo la envolvente
    media ±1 desviación estándar.

    Parámetros
    ----------
    Y_pie : np.ndarray
        Tensor (N, T, 8) con las trayectorias de marcadores del pie.
    var_names : list of str
        Nombres asociados a las columnas de Y_pie.
    out_png : str
        Ruta de salida para la figura.
    """
    N, T, K = Y_pie.shape
    t = np.linspace(0, 100, T)

    fig, axes = plt.subplots(2, 3, figsize=(15, 6), dpi=140, sharex=True)
    fig.suptitle("Trayectorias del pie en Z", fontsize=14)

    order_names = [
        "R_ankle_Z", "R_heel_Z", "R_toe_Z",
        "L_ankle_Z", "L_heel_Z", "L_toe_Z"
    ]

    axes = axes.ravel()
    for ax, name in zip(axes, order_names):
        idx = var_names.index(name)
        _plot_foot_channel_on_ax(ax, t, Y_pie[:, :, idx], name)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[FIG] Guardado foot Z en: {out_png}")


def analyze_gait_events(Y_pie, var_names, printer=print, z_margin=0.02):
    """
    Estima, a partir de las trayectorias verticales de talón y antepié,
    la fracción del ciclo en la que cada pie se encuentra en apoyo, en
    oscilación y en doble apoyo.

    Se considera que un pie está en apoyo si al menos uno de sus puntos
    (talón o antepié) se encuentra por debajo de un margen respecto al
    plano de apoyo (z = 0).

    Parámetros
    ----------
    Y_pie : np.ndarray
        Tensor (N, T, 8) con las trayectorias del pie.
    var_names : list of str
        Lista de nombres de los canales en Y_pie.
    printer : callable
        Función de impresión para registrar el resumen.
    z_margin : float
        Margen vertical (en metros) utilizado para decidir el contacto
        con el suelo.

    Salida
    ------
    Imprime un resumen estadístico (media, desviación típica y percentiles)
    de:
        - Duración de apoyo (stance) de cada pie.
        - Duración de oscilación (swing) de cada pie.
        - Duración de doble apoyo.
    """
    N, T, K = Y_pie.shape

    i_Rheel = var_names.index("R_heel_Z")
    i_Rtoe = var_names.index("R_toe_Z")
    i_Lheel = var_names.index("L_heel_Z")
    i_Ltoe = var_names.index("L_toe_Z")

    Z_Rh = Y_pie[:, :, i_Rheel]
    Z_Rt = Y_pie[:, :, i_Rtoe]
    Z_Lh = Y_pie[:, :, i_Lheel]
    Z_Lt = Y_pie[:, :, i_Ltoe]

    floor_R = np.zeros((N, 1), dtype=np.float32)
    floor_L = np.zeros((N, 1), dtype=np.float32)

    # Pie en apoyo si talón o antepié están próximos al suelo
    stance_R = (Z_Rh <= floor_R + z_margin) | (Z_Rt <= floor_R + z_margin)
    stance_L = (Z_Lh <= floor_L + z_margin) | (Z_Lt <= floor_L + z_margin)

    swing_R = ~stance_R
    swing_L = ~stance_L

    stance_R_pct = np.zeros(N)
    stance_L_pct = np.zeros(N)
    swing_R_pct = np.zeros(N)
    swing_L_pct = np.zeros(N)
    ds_pct = np.zeros(N)

    for i in range(N):
        sR = stance_R[i, :]
        sL = stance_L[i, :]

        stance_R_pct[i] = 100.0 * np.mean(sR)
        stance_L_pct[i] = 100.0 * np.mean(sL)

        swing_R_pct[i] = 100.0 * np.mean(~sR)
        swing_L_pct[i] = 100.0 * np.mean(~sL)

        ds = sR & sL
        ds_pct[i] = 100.0 * np.mean(ds)

    def nan_stats(x):
        return np.nanmean(x), np.nanstd(x), np.nanpercentile(x, [5, 50, 95])

    mR_st, sR_st, pR_st = nan_stats(stance_R_pct)
    mL_st, sL_st, pL_st = nan_stats(stance_L_pct)
    mR_sw, sR_sw, pR_sw = nan_stats(swing_R_pct)
    mL_sw, sL_sw, pL_sw = nan_stats(swing_L_pct)
    mDS, sDS, pDS = nan_stats(ds_pct)

    printer("\n========== RESUMEN EVENTOS DE APOYO (modelo cinemático) ==========\n")

    printer(">> Duración de apoyo (stance) por ciclo:")
    printer(
        f"   Pie derecho:  media = {mR_st:5.1f}%  (std = {sR_st:4.1f}%)   "
        f"p5/p50/p95 = {pR_st[0]:4.1f} / {pR_st[1]:4.1f} / {pR_st[2]:4.1f}%"
    )
    printer(
        f"   Pie izquierdo: media = {mL_st:5.1f}%  (std = {sL_st:4.1f}%)   "
        f"p5/p50/p95 = {pL_st[0]:4.1f} / {pL_st[1]:4.1f} / {pL_st[2]:4.1f}%\n"
    )

    printer(">> Duración de oscilación (swing) por ciclo:")
    printer(
        f"   Pie derecho:  media = {mR_sw:5.1f}%  (std = {sR_sw:4.1f}%)   "
        f"p5/p50/p95 = {pR_sw[0]:4.1f} / {pR_sw[1]:4.1f} / {pR_sw[2]:4.1f}%"
    )
    printer(
        f"   Pie izquierdo: media = {mL_sw:5.1f}%  (std = {sL_sw:4.1f}%)   "
        f"p5/p50/p95 = {pL_sw[0]:4.1f} / {pL_sw[1]:4.1f} / {pL_sw[2]:4.1f}%\n"
    )

    printer(">> Doble apoyo (double support) por ciclo:")
    printer(
        f"   media = {mDS:5.1f}%  (std = {sDS:4.1f}%)   "
        f"p5/p50/p95 = {pDS[0]:4.1f} / {pDS[1]:4.1f} / {pDS[2]:4.1f}%\n"
    )

    printer("===================================================================\n")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    """
    Punto de entrada principal del script.

    Funcionalidad global:
        - Carga un archivo .mat producido por VALID_CLINICA.
        - Aplica filtros de calidad y de outliers sobre las condiciones
          clínicas (L,H) y sus valores reconstruidos.
        - Calcula estadísticas de error entre X_cond y X_real.
        - Genera figuras de:
            * Plano L–H (condición vs cinemática).
            * Gráfico L_cond/H_cond frente a L_real/H_real.
            * Quesitos de distribución de error absoluto.
            * Boxplots de error firmado en función de la condición.
            * Trayectorias del pie en X y Z.
        - Calcula y resume las duraciones de apoyo, oscilación y doble apoyo.

    El resultado se vuelca tanto en stdout como en un fichero de informe
    de texto (clinical_report.txt) dentro de la carpeta de salida.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Comparación de L,H entre X_cond y X_real (VALID_CLINICA) "
            "y visualización de trayectorias de pie."
        )
    )
    parser.add_argument(
        "--mat", type=str, required=True,
        help="Ruta al .mat generado por VALID_CLINICA (p.ej. results_db_synth.mat)"
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefijo opcional para las figuras (p.ej. 'synth_run1_')"
    )
    parser.add_argument(
        "--out", type=str, default="eval_LH",
        help="Carpeta donde se guardan las figuras (por defecto: eval_LH)"
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    report_path = os.path.join(args.out, "clinical_report.txt")
    with open(report_path, "w", encoding="utf-8") as rep:

        def report_print(*a, **k):
            """
            Función auxiliar para duplicar la salida tanto en consola como
            en el archivo de informe.
            """
            print(*a, **k)
            print(*a, **k, file=rep)

        report_print("INFORME CLÍNICO DEL DATASET")
        report_print(f"Archivo .mat: {args.mat}")
        if args.prefix:
            report_print(f"Prefijo de sesión: {args.prefix}")
        report_print("")

        # ----------------- Parte L,H: carga + filtros de calidad -----------------
        X_real, X_cond = load_LH_from_mat(args.mat, printer=report_print)

        suf = f"({args.prefix})" if args.prefix else ""
        f_LH = os.path.join(args.out, f"{args.prefix}LH_plane.png")
        plot_LH_plane(X_cond, X_real, out_png=f_LH, title_suffix=suf)

        f_LH_id = os.path.join(args.out, f"{args.prefix}LH_identity.png")
        plot_LH_identity(X_cond, X_real, out_png=f_LH_id, title_suffix=suf)
        report_print(f"[OK] Figura L/H vs L_real/H_real guardada en: {f_LH_id}")

        # Filtro adicional de outliers para las estadísticas
        X_cond_clean, X_real_clean, mask_keep = filter_outliers_LH(
            X_cond, X_real, printer=report_print
        )

        # Estadísticas de concordancia L/H (sin outliers)
        dL, dH = compute_stats(X_cond_clean, X_real_clean, printer=report_print)

        # Quesitos y boxplots de error de condición
        plot_error_and_value_distributions(
            X_cond_clean, X_real_clean, dL, dH,
            out_dir=args.out, prefix=args.prefix, printer=report_print, n_bins=3
        )

        # ----------------- Parte pies: trayectorias + eventos -----------------
        Y_pie = None
        var_names = None
        try:
            Y_pie, var_names = load_feet_from_mat(args.mat)
            f_footX = os.path.join(args.out, f"{args.prefix}foot_X.png")
            f_footZ = os.path.join(args.out, f"{args.prefix}foot_Z.png")
            plot_feet_X(Y_pie, var_names, out_png=f_footX)
            plot_feet_Z(Y_pie, var_names, out_png=f_footZ)
            report_print("[OK] Figuras de trayectorias de pie (X y Z) generadas.")
        except Exception as e:
            msg = f"[WARN] No se pudo generar la figura de pie: {e}"
            print(msg)
            print(msg, file=rep)

        if Y_pie is not None and var_names is not None:
            analyze_gait_events(Y_pie, var_names, printer=report_print)
        else:
            report_print("No se pudo analizar eventos de apoyo: Y_pie no disponible.\n")

    print(f"[OK] Figuras guardadas en la carpeta: {args.out}")
    print(f"[OK] Informe clínico guardado en: {report_path}")


if __name__ == "__main__":
    main()
