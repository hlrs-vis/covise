/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   PDB High detailed definition for how transparency should
//                be calculated
//
// Author:        Philip Weber
//
// Creation Date: 2006-02-29
//
// **************************************************************************

#ifndef HIGH_VISITOR_H
#define HIGH_VISITOR_H

#include "TransparentVisitor.h"

class HighDetailTransVisitor : public TransparentVisitor
{
private:
    virtual void calculateAlphaAndBin(float);

public:
    HighDetailTransVisitor();
};
#endif
