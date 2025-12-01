function [L, H, lengthStep, heightStep, stepAng, foot8] = LH_from_Y_single(Y, S_meas)
% ENTRADA:
%   Y      : 4 x T  [Hip_flex; Knee_flex; Ankle_flex; Pelvis_List]
%   S_meas : 6 x 1  [Height; femur; tibia; Leg_length; KneeWidth; widthPelvis]
%
% SALIDA:
%   L, H        : longitud y altura de paso (m)
%   lengthStep  : idem L
%   heightStep  : struct devuelto por executeKinematicModel
%   stepAng     : struct de ángulos utilizado
%   foot8       : 8 x T con:
%                 [R_ankle_X; R_ankle_Z; R_heel_Z; R_toe_Z;
%                  L_ankle_X; L_ankle_Z; L_heel_Z; L_toe_Z]

    % --- measurements desde S_meas ---
    measurements.bodyheight  = S_meas(1);  % m
    measurements.femur       = S_meas(2);  % m
    measurements.tibia       = S_meas(3);  % m
    measurements.widthPelvis = S_meas(6);  % m

    % Parámetros genéricos (igual que antes)
    measurements.heightAnkle    = 0.08;   % 8 cm
    measurements.distanceToHeel = 0.08;   % 8 cm
    measurements.distanceToToe  = 0.20;   % 20 cm
    measurements.depthPelvis    = 0;

    % --- Construir stepAng directamente desde Y (GAN ya normalizado) ---
    T = size(Y, 2);

    stepAng.hipL_flex   = Y(1, :);
    stepAng.kneeL_flex  = Y(2, :);
    stepAng.ankleL_flex = Y(3, :);
    stepAng.pelvisList  = Y(4, :);

    % fase normalizada 0–100 %
    stepAng.phase = linspace(0, 100, T);

    % --- Añadir pierna contralateral desfasada 50% ---
    stepAng = add_contralateral_50(stepAng);

    % =====================================================================
    % RELLENAR DOFs FALTANTES A CERO PARA QUE executeKinematicModel NO FALLE
    % =====================================================================
    z = zeros(1, T);

    % Abducciones de cadera
    if ~isfield(stepAng, 'hipL_abd'), stepAng.hipL_abd = z; end
    if ~isfield(stepAng, 'hipR_abd'), stepAng.hipR_abd = z; end

    % Rotaciones de cadera
    if ~isfield(stepAng, 'hipL_rot'), stepAng.hipL_rot = z; end
    if ~isfield(stepAng, 'hipR_rot'), stepAng.hipR_rot = z; end

    % Otras DOFs típicas de pelvis (si tu modelo las usa)
    if ~isfield(stepAng, 'pelvisTilt'), stepAng.pelvisTilt = z; end
    if ~isfield(stepAng, 'pelvisRot'),  stepAng.pelvisRot  = z; end

    % (Si más adelante te sale un error de otro campo tipo kneeR_rot,
    %  ankleL_sup, etc., lo añades aquí igual, a cero.)

    % --- Modelo cinemático ---
    % [lengthStep, heightStep, swingPercent, pelvisPos, hipPos, footPos, ...]
    [lengthStep, heightStep, ~, ~, ~, footPos] = ...
        executeKinematicModel(stepAng, measurements, 0);

    L = lengthStep;
    H = heightStep.value;

    % --- Construir matriz 8 x T con posiciones del pie ---
    R_ankle_X = footPos.R.ankle.X(:)'; 
    R_ankle_Z = footPos.R.ankle.Z(:)';
    R_heel_Z  = footPos.R.heel.Z(:)';
    R_toe_Z   = footPos.R.toe.Z(:)';

    L_ankle_X = footPos.L.ankle.X(:)';
    L_ankle_Z = footPos.L.ankle.Z(:)';
    L_heel_Z  = footPos.L.heel.Z(:)';
    L_toe_Z   = footPos.L.toe.Z(:)';

    foot8 = [
        R_ankle_X;
        R_ankle_Z;
        R_heel_Z;
        R_toe_Z;
        L_ankle_X;
        L_ankle_Z;
        L_heel_Z;
        L_toe_Z
    ];
end

