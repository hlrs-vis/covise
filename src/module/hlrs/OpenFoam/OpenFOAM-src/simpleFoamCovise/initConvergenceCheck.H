/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// initialize values for convergence checks

scalar eqnResidual = 1, maxResidual = 0;
scalar convergenceCriterion = 0;

simple.readIfPresent("convergence", convergenceCriterion);
