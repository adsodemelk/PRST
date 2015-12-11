clf
[x, y] = meshgrid(0:10, 0:8);
x(2:8, 2:10) = x(2:8, 2:10) + .25 * randn(7,9);
y(2:8, 2:10) = y(2:8, 2:10) + .25 * randn(7,9);

G = triangleGrid([x(:) y(:)]);
V = pebi(G);
plotCellData(V, diff(V.cells.facePos));
hcb = colorbar
set(hcb, 'YTick', [1 2 3 4 5 6 7 8 9 10])

V = makeLayeredGrid(V, 3);
plotCellData(V, diff(V.cells.facePos));
V.cells = rmfield(V.cells, 'indexMap')
save('grid_equal_without_indexMap.mat', 'V', '-v7')
