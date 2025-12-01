function stepAng = add_contralateral_50(stepAng)
% add_contralateral_50
%   Genera los ángulos de la pierna contralateral (derecha) a partir
%   de la pierna base (izquierda) aplicando un desfase del 50 % del ciclo.
%
%   ENTRADA:
%       stepAng.phase       (1 x N)
%       stepAng.hipL_flex   (1 x N)
%       stepAng.kneeL_flex  (1 x N)
%       stepAng.ankleL_flex (1 x N)
%
%   SALIDA:
%       stepAng.hipR_flex   (1 x N)
%       stepAng.kneeR_flex  (1 x N)
%       stepAng.ankleR_flex (1 x N)

    n    = numel(stepAng.phase);
    half = round(n/2);   % para N=100 -> 50 muestras

    % Desplazamiento circular de media longitud de ciclo
    shift_half = @(x)[x(half+1:end), x(1:half)];

    stepAng.hipR_flex   = shift_half(stepAng.hipL_flex);
    stepAng.kneeR_flex  = shift_half(stepAng.kneeL_flex);
    stepAng.ankleR_flex = shift_half(stepAng.ankleL_flex);
end
