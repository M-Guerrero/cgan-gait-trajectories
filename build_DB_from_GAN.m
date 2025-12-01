%% build_DB_from_GAN.m
%   Construye una BD tipo GAN a partir de las secuencias generadas.
%   Entrada (HDF5, creado en Python):
%       Y_seq  : (N, T, 4) o equivalente permutable
%       S_meas : (N, 6)    o (6, N)
%       X_cond : (N, C)    o (C, N) donde C>=2 (L,H + medidas)
%   Salida (outFile):
%       Y_seq  : 4 x T x N
%       S_meas : 6 x N
%       X_cond : 2 x N (L,H originales)
%       X_real : 2 x N (L,H cinemáticos)
%       Y_pie  : 8 x T x N
%       simIniFin : 1 x N (similitud inicio/fin de ciclo)

clear; clc; close all;

%% Rutas

currentPath   = fileparts(mfilename('fullpath'));
functionsPath = fullfile(currentPath, '..', 'FUNCIONES');
addpath(functionsPath);

databasePath  = fullfile(currentPath, '..', 'DATABASE');
addpath(databasePath);

%% Archivos

syntheticFile = 'db_validacion.mat';          % HDF5 de Python
outFile       = 'db_validacion_w_foot.mat';   % salida v7.3

%% Lectura básica del HDF5

info  = h5info(syntheticFile);
names = {info.Datasets.Name};

if ~ismember('Y_seq', names)
    error('En %s no existe el dataset Y_seq', syntheticFile);
end
if ~ismember('S_meas', names)
    error('En %s no existe el dataset S_meas', syntheticFile);
end

Y_seq_raw  = h5read(syntheticFile, '/Y_seq');
S_meas_raw = h5read(syntheticFile, '/S_meas');

if ismember('X_cond', names)
    X_cond_raw = h5read(syntheticFile, '/X_cond');
    tieneXcond = true;
else
    warning(['En el archivo no existe X_cond. ', ...
             'Solo se guardarán Y_seq, S_meas, X_real y Y_pie.']);
    tieneXcond = false;
end

%% Normalizar S_meas a (6 x N)

szS = size(S_meas_raw);
if numel(szS) ~= 2
    error('S_meas debe ser 2D, pero tiene size=%s', mat2str(szS));
end

idxMeas = find(szS == 6);
if numel(idxMeas) ~= 1
    error('No se identifica una única dimensión de medidas (6) en S_meas, size=%s', mat2str(szS));
end

if idxMeas == 1
    S_meas   = S_meas_raw;
    N_from_S = szS(2);
else
    S_meas   = S_meas_raw.';  % N x 6 -> 6 x N
    N_from_S = szS(1);
end

%% Normalizar X_cond a (2 x N) si existe

if tieneXcond
    szX = size(X_cond_raw);
    if numel(szX) ~= 2
        error('X_cond debe ser 2D, size=%s', mat2str(szX));
    end

    idxFeat = find(szX == 2);
    if numel(idxFeat) ~= 1
        error('No se identifica dimensión (2) en X_cond, size=%s', mat2str(szX));
    end

    if idxFeat == 1
        X_cond   = X_cond_raw;
        N_from_X = szX(2);
    else
        X_cond   = X_cond_raw.';   % N x 2 -> 2 x N
        N_from_X = szX(1);
    end

    if N_from_X ~= N_from_S
        warning('N en X_cond (%d) ≠ N en S_meas (%d). Se toma S_meas como referencia.', ...
                 N_from_X, N_from_S);
    end
end

%% Normalizar Y_seq a (4 x T x N)

szY = size(Y_seq_raw);
if numel(szY) ~= 3
    error('Y_seq debe ser 3D, size=%s', mat2str(szY));
end

% Dimensión de articulaciones (4)
idxJ = find(szY == 4);
if numel(idxJ) ~= 1
    error('No se identifica dimensión (4) en Y_seq, size=%s', mat2str(szY));
end

% Dimensión de muestras N (usa N_from_S como referencia)
candDims = setdiff(1:3, idxJ);
maskN    = (szY(candDims) == N_from_S);
if sum(maskN) ~= 1
    error('No se puede asignar dimensión N=%d en Y_seq, size=%s', N_from_S, mat2str(szY));
end
idxN = candDims(maskN);

% La restante es la dimensión temporal
idxT = setdiff(1:3, [idxJ, idxN]);

% Reordenar a (4 x T x N)
permOrder = [idxJ, idxT, idxN];
Y_seq = permute(Y_seq_raw, permOrder);

T = size(Y_seq, 2);
N = size(Y_seq, 3);

%% Verificación rápida de permutación

idxRaw = cell(1,3);
idxRaw{idxJ} = 1;
idxRaw{idxN} = 1;
idxRaw{idxT} = 1:szY(idxT);

traj_raw = squeeze(Y_seq_raw(idxRaw{:}));
traj_reo = squeeze(Y_seq(1, :, 1));

if isequal(size(traj_raw), size(traj_reo))
    maxDiff = max(abs(traj_raw(:) - traj_reo(:)));
    fprintf('Check permute primer ciclo/articulación: maxDiff = %.3e\n', maxDiff);
else
    warning('La trayectoria raw y la reordenada no coinciden en tamaño.');
end

fprintf('Cargado %s\n', syntheticFile);
fprintf('Y_seq_raw size: %s\n', mat2str(szY));
fprintf('S_meas (6 x N): %s\n', mat2str(size(S_meas)));
if tieneXcond
    fprintf('X_cond (2 x N): %s\n', mat2str(size(X_cond)));
end

%% Cálculo de L/H y pies mediante modelo cinemático

[L_all, H_all, Y_pie] = LH_from_Yseq(Y_seq, S_meas);   % usa LH_from_Y_single dentro
X_real = [L_all; H_all];

fprintf('X_real (L): min=%.3f max=%.3f\n', min(L_all), max(L_all));
fprintf('X_real (H): min=%.3f max=%.3f\n', min(H_all), max(H_all));
fprintf('Y_pie size: %s\n', mat2str(size(Y_pie)));

%% Filtrado QA por similitud inicio/fin de ciclo

thrSim    = 95;             % umbral mínimo (%)
simIniFin = zeros(1, N);

for i = 1:N
    y_ini = Y_seq(:, 1,   i);
    y_fin = Y_seq(:, end, i);

    num = norm(y_fin - y_ini);
    den = max(norm(y_ini), eps);
    simIniFin(i) = max(0, 1 - num/den) * 100;
end

goodMask = simIniFin >= thrSim;
N_good   = sum(goodMask);

fprintf('Filtrado QA inicio/fin: %d de %d (%.1f%%) con similitud >= %.1f%%\n', ...
        N_good, N, 100*N_good/N, thrSim);

Y_seq  = Y_seq(:, :,  goodMask);
S_meas = S_meas(:,   goodMask);
X_real = X_real(:,   goodMask);
Y_pie  = Y_pie(:, :, goodMask);
if tieneXcond
    X_cond = X_cond(:, goodMask);
end
N = N_good;

%% Guardado

if tieneXcond
    save(outFile, 'Y_seq', 'S_meas', 'X_cond', 'X_real', 'Y_pie', 'simIniFin', '-v7.3');
else
    save(outFile, 'Y_seq', 'S_meas', 'X_real', 'Y_pie', 'simIniFin', '-v7.3');
end

fprintf('Guardado: %s (v7.3)\n', outFile);

