try
    require incomp
catch %#ok<CTCH>
    mrstModule add incomp
end

gravity reset on
G          = cartGrid([1, 1, 30], [1, 1, 30]);
G          = computeGeometry(G);
rock       = makeRock(G, 0.1*darcy, 1);
fluid      = initSingleFluid('mu' ,    1*centi*poise, ...
                             'rho', 1014*kilogram/meter^3);
bc  = pside([], G, 'TOP', 100.*barsa());
T   = computeTrans(G, rock);
sol = incompTPFA(initResSol(G, 0.0), G, T, fluid, 'bc', bc);
save('sol.mat', 'sol' ,'-v7')
