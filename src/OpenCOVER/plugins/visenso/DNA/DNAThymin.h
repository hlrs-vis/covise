/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _THYMIN_H
#define _THYMIN_H

#include "DNABase.h"
class DNABaseUnitConnectionPoint;

class DNAThymin : public DNABase
{
public:
    DNAThymin(osg::Matrix m, int num = 0);
    virtual ~DNAThymin();

    // set Connection form Gui
    virtual void setConnection(std::string nameConnPoint, std::string nameConnPoint2, bool connected, bool enabled, DNABaseUnit *connObj, bool sendBack = true);
    // reposition unit to dock to the other unit
    virtual bool connectTo(DNABaseUnit *otherUnit, DNABaseUnitConnectionPoint *myConnectionPoint = NULL, DNABaseUnitConnectionPoint *otherConnectionPoint = NULL);
    // check if connectionpoint is free (need for adenin, thymin, cytosin, guanin)
    virtual bool isConnectionPossible(std::string connPoint);

    void enableOtherConnPoints(DNABaseUnitConnectionPoint *myConnectionPoint, DNABaseUnitConnectionPoint *otherConnectionPoint, bool connected, bool callConnectedPoint = true);

private:
    DNABaseUnitConnectionPoint *a1;
    DNABaseUnitConnectionPoint *g1;
    DNABaseUnitConnectionPoint *c1;
};

#endif
