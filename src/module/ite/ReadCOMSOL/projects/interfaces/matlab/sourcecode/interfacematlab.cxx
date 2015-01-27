/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "../include/interfacematlab.hxx"
#include "interfacematlabcom.hxx"
#include "interfacematlabengine.hxx"
#include <stdlib.h>

InterfaceMatlab::InterfaceMatlab(void)
{
}

InterfaceMatlab::~InterfaceMatlab()
{
}

InterfaceMatlab *InterfaceMatlab::getInstance()
{
    return new InterfaceMatlabCom();
    // return new InterfaceMatlabEngine();
}
