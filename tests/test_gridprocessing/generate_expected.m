%% tensorGrid


% A very simple grid
clear
G = tensorGrid([0 1 2], [0 1 2]);
save('expected_tensorGrid2D_1.mat', 'G', '-v7')


% with depthz
G = tensorGrid([1 2 3], [0.5, 1, 1.5], [10, 20], 'depthz', [
    1 2 3;
    4 5 6;
    7 8 9]);
save('expected_tensorGrid3D_1.mat', 'G', '-v7')


% without depthz
G = tensorGrid([1 2 3], [0.5, 1, 1.5], [10, 20]);
save('expected_tensorGrid3D_2.mat', 'G', '-v7')


%% cartGrid

G = cartGrid([3 5], [1 1]);
save('cartGrid2D_simple.mat', 'G', '-v7')

G = cartGrid([3 5 7], [1 1 3]);
save('cartGrid3D_simple.mat', 'G', '-v7')


disp('Expected data created.')