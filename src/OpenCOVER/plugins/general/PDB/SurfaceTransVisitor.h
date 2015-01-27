/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   PDB Low detailed definition for how transparency should
//		  be calculated
//
// Author:        Philip Weber
//
// Creation Date: 2006-02-29
//
// **************************************************************************

#ifndef SUR_VISITOR_H
#define SUR_VISITOR_H

#include "TransparentVisitor.h"

class SurfaceTransVisitor : public TransparentVisitor
{
private:
    virtual void calculateAlphaAndBin(float); // need to make pure virtual

public:
    SurfaceTransVisitor();
};
#endif
