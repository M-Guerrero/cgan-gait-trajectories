function stepAng = build_stepAng_from_Y(Y, heel_idx, numPoints)
% build_stepAng_from_Y
%   Normaliza un ciclo articular a un número fijo de muestras y define
%   los campos básicos de stepAng para la pierna base.
%
%   ENTRADA:
%       Y        : 4 x T
%                  [Hip_flex; Knee_flex; Ankle_flex; Pelvis_List]
%       heel_idx : índice de frame asociado a heel strike (0 % ciclo)
%       numPoints: número de muestras del ciclo normalizado (p.ej. 100)
%
%   SALIDA:
%       stepAng.phase       (1 x numPoints)
%       stepAng.hipL_flex   (1 x numPoints)
%       stepAng.kneeL_flex  (1 x numPoints)
%       stepAng.ankleL_flex (1 x numPoints)
%       stepAng.pelvisList  (1 x numPoints)
%       stepAng.pelvisTilt  (1 x numPoints, fijado a cero)
%       stepAng.hipL_abd    (1 x numPoints, fijado a cero)
%       stepAng.hipR_abd    (1 x numPoints, fijado a cero)

    if nargin < 3 || isempty(numPoints)
        numPoints = 100;
    end
    if nargin < 2 || isempty(heel_idx)
        heel_idx = 1;
    end

    hip    = Y(1, :);
    knee   = Y(2, :);
    ankle  = Y(3, :);
    pelvis = Y(4, :);

    T          = numel(hip);
    phase_raw  = linspace(0, 100, T);

    % 1) Recentrado del ciclo para que heel_idx corresponda al 0 %
    shift  = 1 - heel_idx;   % si heel_idx = 1, no se desplaza
    hip    = circshift(hip,    [0, shift]);
    knee   = circshift(knee,   [0, shift]);
    ankle  = circshift(ankle,  [0, shift]);
    pelvis = circshift(pelvis, [0, shift]);

    % 2) Remuestreo uniforme a numPoints (0–100 %)
    phase_new = linspace(0, 100, numPoints);
    hip_n     = interp1(phase_raw, hip,    phase_new, 'pchip');
    knee_n    = interp1(phase_raw, knee,   phase_new, 'pchip');
    ankle_n   = interp1(phase_raw, ankle,  phase_new, 'pchip');
    pelvis_n  = interp1(phase_raw, pelvis, phase_new, 'pchip');

    % 3) Forzar periodicidad (última muestra = primera)
    hip_n(end)    = hip_n(1);
    knee_n(end)   = knee_n(1);
    ankle_n(end)  = ankle_n(1);
    pelvis_n(end) = pelvis_n(1);

    % 4) Construcción de la estructura stepAng (pierna izquierda como base)
    stepAng.phase       = phase_new;
    stepAng.hipL_flex   = hip_n;
    stepAng.kneeL_flex  = knee_n;
    stepAng.ankleL_flex = ankle_n;
    stepAng.pelvisList  = pelvis_n;

    % Grados de libertad no utilizados en el modelo -> cero
    stepAng.pelvisTilt  = zeros(1, numPoints);
    stepAng.hipL_abd    = zeros(1, numPoints);
    stepAng.hipR_abd    = zeros(1, numPoints);
end

