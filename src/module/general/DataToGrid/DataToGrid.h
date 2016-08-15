/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                           (C)2005 ZAIK **
**                                                                        **
** Description: Transform data to a grid                                  **
**                                                                        **
**      Author: Martin Aumueller (aumueller@uni-koeln.de)                 **
**                                                                        **
\**************************************************************************/

#ifndef _DATATOGRID_H_
#define _DATATOGRID_H_

#include <api/coSimpleModule.h>
using namespace covise;

class DataToGrid : public coSimpleModule
{
private:
    // Ports:
    coInputPort *piData;
    coInputPort *piTopGrid;
    coOutputPort *poGrid;

    // Parameters:
    coChoiceParam *pcDataDirection;
    coBooleanParam *pbStrGrid;

    coIntScalarParam *psResX;
    coIntScalarParam *psResY;
    coIntScalarParam *psResZ;
    coFloatParam *psSizeX;
    coFloatParam *psSizeY;
    coFloatParam *psSizeZ;

    // Methods:
    virtual int compute(const char *port);

public:
    DataToGrid(int argc, char *argv[]);
};

#endif
