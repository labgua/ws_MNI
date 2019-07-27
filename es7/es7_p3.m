
%%% es7_p3
addpath('/home/sergio/ws_mni/ws_MNI/mathlab/');
 
t = [0;0;0;0;1;2;3;4;5;5;6;7];
P = [-2.5 -8; 0 3; 1 -2; 5 -7.5; 10 4; 11 6; 15 4; 22.5 7.5];
plot(P(:, 1), P(:, 2), '*r', 'DisplayName', 'Punti di controllo')
hold on
line(P(:, 1), P(:, 2), 'Color', 'black', 'LineStyle', '--', 'DisplayName', 'Poligono di controllo')
[C, U] = bspline_deboor(4, t, transpose(P));
plot(C(1, :), C(2, :), 'b', 'DisplayName', 'B-Spline cubica')
legend('Location', 'southeast')
hold off
