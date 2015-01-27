/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Solve the Momentum equation

tmp<fvVectorMatrix> UEqn(
    fvm::div(phi, U)
    + turbulence->divDevReff(U));

UEqn().relax();

eqnResidual = solve(
                  UEqn() == -fvc::grad(p)).initialResidual();

maxResidual = max(eqnResidual, maxResidual);
