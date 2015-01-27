/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

p.boundaryField().updateCoeffs();

volScalarField AU = UEqn().A();
U = UEqn().H() / AU;
UEqn.clear();
phi = fvc::interpolate(U) & mesh.Sf();
adjustPhi(phi, U, p);

// Non-orthogonal pressure corrector loop
for (int nonOrth = 0; nonOrth <= nNonOrthCorr; nonOrth++)
{
    fvScalarMatrix pEqn(
        fvm::laplacian(1.0 / AU, p) == fvc::div(phi));

    pEqn.setReference(pRefCell, pRefValue);

    // Retain the residual from the first iteration
    if (nonOrth == 0)
    {
        eqnResidual = pEqn.solve().initialResidual();
        maxResidual = max(eqnResidual, maxResidual);
    }
    else
    {
        pEqn.solve();
    }

    if (nonOrth == nNonOrthCorr)
    {
        phi -= pEqn.flux();
    }
}

#include "continuityErrs.H"

// Explicitly relax pressure for momentum corrector
p.relax();

// Momentum corrector
U -= fvc::grad(p) / AU;
U.correctBoundaryConditions();
