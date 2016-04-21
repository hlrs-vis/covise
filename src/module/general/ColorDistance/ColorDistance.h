/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                        (C)2005 RRZK  **
 ** Description:   Simple calculations on volume data.                   **
 **                                                                      **
 ** Author:        Martin Aumueller <aumueller@uni-koeln.de>             **
 **                Sandra Meid <sandra.meid@uni-koeln.de>                **
 **                                                                      **
 ** Creation Date: 05.01.2005                                            **
\**************************************************************************/

#ifndef _COLOR_DISTANCE_H
#define _COLOR_DISTANCE_H

#include <api/coSimpleModule.h>
using namespace covise;

class coColorDistance : public coSimpleModule
{
private:
    // Ports:
    coInputPort *piR;
    coInputPort *piG;
    coInputPort *piB;

    coOutputPort *poVolume;

    // Parameters:
    coColorParam *paReferenceColor;
    coFloatSliderParam *paSlider;
    coFloatSliderParam *paSlider2;
    coFloatVectorParam *paMinMax;
    coChoiceParam *paMetric;
    coChoiceParam *paColorSpace;
    //coFloatParam*        pfsIgnoreValue;

    // Methods:
    virtual int compute(const char *port);

public:
    coColorDistance(int argc, char *argv[]);
};
#endif
