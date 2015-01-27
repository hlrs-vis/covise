/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ADENIN_H
#define _ADENIN_H

#include "DNABase.h"
class DNABaseUnitConnectionPoint;

class DNAAdenin : public DNABase
{
public:
    DNAAdenin(osg::Matrix m, int num = 0);
    virtual ~DNAAdenin();

    // set Connection form Gui
    virtual void setConnection(std::string nameConnPoint, std::string nameConnPoint2, bool connected, bool enabled, DNABaseUnit *connObj, bool sendBack = true);
    // reposition unit to dock to the other unit
    virtual bool connectTo(DNABaseUnit *otherUnit, DNABaseUnitConnectionPoint *myConnectionPoint = NULL, DNABaseUnitConnectionPoint *otherConnectionPoint = NULL);
    // check if connectionpoint is free (need for adenin, thymin, cytosin, guanin)
    virtual bool isConnectionPossible(std::string connPoint);

    void enableOtherConnPoints(DNABaseUnitConnectionPoint *myConnectionPoint, DNABaseUnitConnectionPoint *otherConnectionPoint, bool connected, bool callConnectedPoint = true);

private:
    DNABaseUnitConnectionPoint *t1;
    DNABaseUnitConnectionPoint *c1;
    DNABaseUnitConnectionPoint *g1;
};

#endif
