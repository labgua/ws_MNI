
%%% es7_p1
addpath('/home/sergio/ws_mni/ws_MNI/mathlab/');

% crea la curva di B?ier per l'insieme di punti P
P = [2.5 -5; 9 9; 19 -2.5; 22.5 7.5];
plot(P(:, 1), P(:, 2), '*r', 'DisplayName', 'Punti di controllo')
hold on
line(P(:, 1), P(:, 2), 'Color', 'black', 'LineStyle', '--', 'DisplayName', 'Poligono di controllo')

% print labels for control points
p_index = 0;
for i = 1: rows(P)
  text(P(i, 1), P(i, 2) - 0.5, ['P_'  num2str(p_index)]);
  p_index = p_index + 1;
endfor

% draw B?ier curve
[B] = de_Casteljau(P, 0.01);
plot(B(:, 1), B(:, 2), 'b', 'DisplayName', 'Curva di Bezier')

% modifica P2
P(3, 2) = -6;
plot(P(3, 1), P(3, 2), '*m', 'DisplayName', 'Punto di controllo alterato')
text(P(3, 1), P(3, 2) - 0.5, ['P_2' "'"]);
line(P(:, 1), P(:, 2), 'Color', 'black', 'LineStyle', '--')
[B] = de_Casteljau(P, 0.01);


plot(B(:, 1), B(:, 2), 'g', 'DisplayName', 'Curva di Bezier alterata')
legend('Location', 'south')
hold off
