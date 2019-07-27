
%%% es7_p2
addpath('/home/sergio/ws_mni/ws_MNI/mathlab/');

% crea la curva di Bézier per l'insieme di punti P
P = [0 0; 1 3; 3 5; 5 12; 7 4];
plot(P(:, 1), P(:, 2), '*r', 'DisplayName', 'Punti di controllo')
hold on
line(P(:, 1), P(:, 2), 'Color', 'black', 'LineStyle', '--', 'DisplayName', 'Poligono di controllo')
[B] = de_Casteljau(P, 0.01);
plot(B(:, 1), B(:, 2), 'b', 'DisplayName', 'Curva di Bezier')

% scala la curva di Bézier
Phalf = P / 2;
plot(Phalf(:, 1), Phalf(:, 2), '*m', 'DisplayName', 'Punti di controllo scalati')
line(Phalf(:, 1), Phalf(:, 2), 'Color', 'black', 'LineStyle', '--');
[Bhalf] = de_Casteljau(Phalf, 0.01);
plot(Bhalf(:, 1), Bhalf(:, 2), 'g', 'DIsplayName', 'Curva di Bezier scalata')
legend('Location', 'northwest')
hold off
