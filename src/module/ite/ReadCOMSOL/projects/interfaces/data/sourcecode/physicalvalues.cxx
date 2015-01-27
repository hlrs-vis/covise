/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "../include/physicalvalues.hxx"
#include "physicalvaluesstandard.h"

PhysicalValues::PhysicalValues()
{
}

PhysicalValues::~PhysicalValues()
{
}

PhysicalValues *PhysicalValues::getInstance(const unsigned int noTimeSteps, const std::vector<unsigned long> &noEvaluationPoints)
{
    return new PhysicalValuesStandard(noTimeSteps, noEvaluationPoints);
}
