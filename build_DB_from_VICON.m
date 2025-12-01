%% build_DB_from_VICON.m
%   Construye una BD tipo GAN a partir de datos VICON.
%   Salida:
%       Y_seq  : 4 x 100 x N   [Hip_flex; Knee_flex; Ankle_flex; Pelvis_List]
%       S_meas : 6 x N         [Height; femur; tibia; Leg_length; KneeWidth; widthPelvis]
%       X_cond : 2 x N         [L; H] (m)

clear; clc; close all;

%% Rutas

currentPath   = fileparts(mfilename('fullpath'));
databasePath  = fullfile(currentPath, '..', 'DATABASE');
addpath(databasePath);

functionsPath = fullfile(currentPath, '..', 'FUNCIONES');
addpath(functionsPath);

%% Archivos

VICON_file = 'datosVICON_test.mat';
outFile    = 'db_test.mat';

load(VICON_file);   % estructura 'data'

ids = fieldnames(data);

Y_list = {};   % cada entrada: 4 x 100
S_list = [];   % 6 x N
X_list = [];   % 2 x N

nTotal   = 0;
nOK      = 0;
nSkipped = 0;

% Se usan ambas piernas:
patas = {'Normalized_R', 'Normalized_L'};

%% Bucle sujeto / velocidad / longitud / altura / pata / ciclo

for i = 2:numel(ids)   % se empieza en 2 si id1 es problemático
    id = ids{i};

    speedNames = fieldnames(data.(id).Trials);

    for s = 1:numel(speedNames)
        speed_str = speedNames{s};

        longitudes = fieldnames(data.(id).Trials.(speed_str));
        for l = 1:numel(longitudes)
            long_str = longitudes{l};

            % opcional: excluir freeLength
            if strcmp(long_str, 'freeLength')
                continue;
            end

            alturas = fieldnames(data.(id).Trials.(speed_str).(long_str));
            for a = 1:numel(alturas)
                alt_str = alturas{a};

                for p = 1:numel(patas)
                    pata = patas{p};

                    if ~isfield(data.(id).Trials.(speed_str).(long_str).(alt_str), pata)
                        continue;
                    end

                    angles  = data.(id).Trials.(speed_str).(long_str).(alt_str).(pata).Angles;
                    markers = data.(id).Trials.(speed_str).(long_str).(alt_str).(pata).Markers;
                    meas    = data.(id).Measurements;

                    nSteps = size(angles.Hip_flex_L, 2);  % nº de ciclos en esa condición

                    for test = 1:nSteps
                        nTotal = nTotal + 1;

                        fprintf('Subj: %s | %s | %s | %s | %s | test %d/%d\n', ...
                            id, speed_str, long_str, alt_str, pata, test, nSteps);

                        try
                            use_auto_hs = true;  % heel strike automático con LHEE.Z

                            [L, H, stepAng, Y_norm] = LH_from_VICON_angles( ...
                                angles, markers, meas, test, use_auto_hs);

                            % Pierna izquierda -> “derecha equivalente” (cambia signo Pelvis_List)
                            if strcmp(pata, 'Normalized_L')
                                Y_norm(4, :) = -Y_norm(4, :);
                            end

                            % Filtro fisiológico:
                            %   L < 0.80 m, 0.05 <= H <= 0.45 m
                            if (L < 0.80) && (H >= 0.05) && (H <= 0.45)

                                % S_meas = [Height; femur; tibia; Leg_length; KneeWidth; widthPelvis]
                                Height     = meas.Height / 100;     % cm -> m
                                femur      = meas.femur;
                                tibia      = meas.tibia;
                                Leg_length = femur + tibia;
                                if isfield(meas, 'KneeWidth')
                                    KneeWidth = meas.KneeWidth;
                                else
                                    KneeWidth = 0;
                                end
                                widthPelvis = meas.widthPelvis;

                                S_vec = [Height; femur; tibia; Leg_length; KneeWidth; widthPelvis];

                                Y_list{end+1}     = Y_norm;    % 4 x 100
                                S_list(:, end+1)  = S_vec;     % 6 x N
                                X_list(:, end+1)  = [L; H];    % 2 x N

                                nOK = nOK + 1;
                            else
                                fprintf('  -> descartado por rango L/H (L=%.3f, H=%.3f)\n', L, H);
                                nSkipped = nSkipped + 1;
                            end

                        catch ME
                            fprintf('  !! ERROR subj=%s, speed=%s, long=%s, alt=%s, pata=%s, test=%d: %s\n', ...
                                id, speed_str, long_str, alt_str, pata, test, ME.message);
                            nSkipped = nSkipped + 1;
                            continue;
                        end
                    end
                end
            end
        end
    end
end

fprintf('\nTotal ciclos procesados: %d\n', nTotal);
fprintf('Ciclos válidos guardados: %d\n', nOK);
fprintf('Ciclos descartados/error: %d\n', nSkipped);

%% Construcción de Y_seq, S_meas, X_cond

if nOK == 0
    error('No se ha guardado ningún ciclo válido. Revisa filtros y datos.');
end

T = size(Y_list{1}, 2);   % típicamente 100
N = numel(Y_list);

Y_seq = zeros(4, T, N);
for n = 1:N
    Y_seq(:, :, n) = Y_list{n};
end

S_meas = S_list;   % 6 x N
X_cond = X_list;   % 2 x N

fprintf('BD final: Y_seq (%d x %d x %d), S_meas (%d x %d), X_cond (%d x %d)\n', ...
    size(Y_seq,1), size(Y_seq,2), size(Y_seq,3), ...
    size(S_meas,1), size(S_meas,2), ...
    size(X_cond,1), size(X_cond,2));

%% Guardado

save(outFile, 'Y_seq', 'S_meas', 'X_cond', '-v7.3');
fprintf('Guardado: %s (v7.3)\n', outFile);
