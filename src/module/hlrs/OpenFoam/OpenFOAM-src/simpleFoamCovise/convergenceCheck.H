/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// check convergence

if (maxResidual < convergenceCriterion)
{
    Info << "reached convergence criterion: " << convergenceCriterion << endl;
    runTime.writeAndEnd();
    Info << "latestTime = " << runTime.timeName() << endl;
}
