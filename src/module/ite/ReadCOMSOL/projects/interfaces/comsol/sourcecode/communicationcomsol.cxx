/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "../include/communicationcomsol.hxx"
#include "communicationcomsolstandard.h"

CommunicationComsol::CommunicationComsol()
{
}

CommunicationComsol::~CommunicationComsol()
{
}

CommunicationComsol *CommunicationComsol::getInstance(const InterfaceMatlab *matlab)
{
    return new CommunicationComsolStandard(matlab);
}