% 3D extruded triangle grid for testing computeGeometry
sz = 5;
[x, y] = meshgrid(0:sz, 0:sz);
%x(2:8, 2:10) = x(2:8, 2:10) + .25 * randn(7,9);
%y(2:8, 2:10) = y(2:8, 2:10) + .25 * randn(7,9);
%z(2:8, 2:10) = z(2:8, 2:10) + .25 * randn(7,9);
size(x)
x = x + .25 * randn(sz+1, sz+1)
y = y + .25 * randn(sz+1, sz+1)

G = triangleGrid([x(:) y(:)]);
G = makeLayeredGrid(G, 3);
G = computeGeometry(G);
plotGrid(G);
save('computeGeometry_triangleGrid3D.mat' ,'G', '-v7');
%V = pebi(G);
%plotCellData(V, diff(V.cells.facePos));
%hcb = colorbar
%set(hcb, 'YTick', [1 2 3 4 5 6 7 8 9 10])
%V = makeLayeredGrid(V, 3);
%plotCellData(V, diff(V.cells.facePos));
