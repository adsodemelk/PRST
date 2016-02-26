% Stripped version of gravityColumn example used to generate tests answer.
G          = cartGrid([1, 1, 30], [1, 1, 30]);
G          = computeGeometry(G);
rock       = makeRock(G, 0.1*darcy, 1);
rock.perm(1:G.cells.num/2) = 0.2*darcy;
T   = computeTrans(G, rock);
save('computeTrans_gravityColumn_T.mat', 'T')
