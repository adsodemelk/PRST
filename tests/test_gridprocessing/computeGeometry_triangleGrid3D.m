% 3D extruded triangle grid for testing computeGeometry
sz = 5;
[x, y] = meshgrid(0:sz, 0:sz);
x(2:sz,2:sz) = x(2:sz,2:sz) + .25;
y = y + .4;

G = triangleGrid([x(:) y(:)]);
G = makeLayeredGrid(G, 3);

plotGrid(G);
save('computeGeometry_triangleGrid3D.mat' ,'G', '-v7');

G = computeGeometry(G);
save('computeGeometry_triangleGrid3D_expected.mat' ,'G', '-v7');