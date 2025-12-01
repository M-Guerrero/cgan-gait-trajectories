function [L, H, stepAng, Y_norm] = LH_from_VICON_angles(angles, markers, meas, test, use_auto_hs)
% LH_from_VICON_angles
%   Extrae un ciclo de marcha a partir de datos VICON, normaliza el
%   ciclo, construye la pierna contralateral y aplica el modelo
%   cinemático para obtener L y H.
%
%   ENTRADA:
%       angles     : struct con campos
%                       Hip_flex_L, Knee_flex_L, Ankle_flex_L, Pelvis_List
%       markers    : struct con al menos LHEE.Z (talón ipsilateral)
%       meas       : struct de mediciones del sujeto (data.(id).Measurements)
%       test       : índice de ciclo (columna en Angles.*(:, test))
%       use_auto_hs: true -> heel strike automático con LHEE.Z
%                    false -> se toma heel_idx = 1
%
%   SALIDA:
%       L, H    : longitud y altura de paso (m)
%       stepAng : struct de ángulos normalizados (L+R) para executeKinematicModel
%       Y_norm  : 4 x 100 [Hip; Knee; Ankle; Pelvis_List] (pierna base)

    if nargin < 5
        use_auto_hs = true;
    end

    % 1) Ángulos de la pierna base (izquierda en este convenio)
    hipL    = angles.Hip_flex_L(:, test)';    % 1 x T
    kneeL   = angles.Knee_flex_L(:, test)';
    ankleL  = angles.Ankle_flex_L(:, test)';
    pelvisL = angles.Pelvis_List(:, test)';

    Y = [hipL; kneeL; ankleL; pelvisL];       % 4 x T

    % 2) Detección de heel strike (si se activa)
    if use_auto_hs
        heelZ = markers.LHEE.Z(:, test);      % talón pierna base
        [~, heel_idx] = min(heelZ);           % frame de mínima altura
    else
        heel_idx = 1;                         % se asume ya alineado
    end

    % 3) Normalización del ciclo a numPoints muestras
    numPoints = 100;
    stepAng = build_stepAng_from_Y(Y, heel_idx, numPoints);

    % 4) Añadir pierna contralateral con desfase del 50 % del ciclo
    stepAng = add_contralateral_50(stepAng);

    % 5) Construcción de measurements para el modelo cinemático
    measurements.bodyheight     = meas.Height / 100;   % cm -> m
    measurements.femur          = meas.femur;
    measurements.tibia          = meas.tibia;
    measurements.widthPelvis    = meas.widthPelvis;
    measurements.heightAnkle    = meas.heightAnkle;
    measurements.distanceToHeel = meas.distanceToHeel;
    measurements.distanceToToe  = meas.distanceToToe;
    measurements.depthPelvis    = 0;                  % si no se usa, se fija a 0

    % 6) Modelo cinemático: genera L y H a partir de stepAng + measurements
    [lengthStep, heightStep, ~, ~, ~, ~] = ...
        executeKinematicModel(stepAng, measurements, 0);

    L = lengthStep;
    H = heightStep.value;

    % 7) Salida de cinemática normalizada de la pierna base
    Y_norm = [stepAng.hipL_flex;
              stepAng.kneeL_flex;
              stepAng.ankleL_flex;
              stepAng.pelvisList];
end
