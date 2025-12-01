function [L_all, H_all, Y_pie] = LH_from_Yseq(Y_seq, S_meas)
% LH_from_YSEQ
%   Aplica el modelo cinemático ciclo a ciclo para obtener longitud
%   y altura de paso, junto con las trayectorias del pie.
%
%   ENTRADA:
%       Y_seq  : 4 x T x N   (ciclos articulares)
%       S_meas : 6 x N       (medidas antropométricas)
%
%   SALIDA:
%       L_all  : 1 x N       (longitud de paso, m)
%       H_all  : 1 x N       (altura de paso, m)
%       Y_pie  : 8 x T x N   (trayectorias del pie)

    N = size(Y_seq, 3);
    T = size(Y_seq, 2);

    L_all = zeros(1, N);
    H_all = zeros(1, N);
    Y_pie = zeros(8, T, N);

    % Envoltorio sobre LH_from_Y_single (procesa un único ciclo)
    for n = 1:N
        Yn = squeeze(Y_seq(:, :, n));  % 4 x T
        Sm = S_meas(:, n);             % 6 x 1

        [L_all(n), H_all(n), ~, ~, ~, foot8] = LH_from_Y_single(Yn, Sm);
        Y_pie(:, :, n) = foot8;        % 8 x T
    end
end
