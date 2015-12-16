% 2D triangle grid for testing _findHeighbors
[x, y] = meshgrid(0:10, 0:8);
x(2:8, 2:10) = x(2:8, 2:10) + .35;
y(2:8, 2:10) = y(2:8, 2:10) + .20;

G = triangleGrid([x(:) y(:)]);
G.faces = rmfield(G.faces, 'neighbors');
save('computeGeometry_findNeighbors2D.mat' ,'G', '-v7');