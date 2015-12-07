
% A very simple grid
clear
G = tensorGrid([0 1 2], [0 1 2]);
save('expected_tensorGrid2D_1G.mat', 'G', '-v7')

V = tensorGrid([0 1 2], [0 1 2]);
save('expected_tensorGrid2D_1V.mat', 'V', '-v7')

disp('Expected data created.')