
% A very simple grid
clear
G = tensorGrid([0 1 2], [0 1 2]);
save('expected_tensorGrid2D_1.mat', 'G', '-v7')

disp('Expected data created.')