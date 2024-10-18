/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeQuadSet.h

#ifndef _VRMLNODEQUADSET_
#define _VRMLNODEQUADSET_

#include "VrmlNodePolygonsCommon.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeQuadSet : public VrmlNodePolygonsCommon

{

public:
    // Define the fields of indexed face set nodes
  static void initFields(VrmlNodeQuadSet *node, VrmlNodeType *t);
  static const char *name();

  VrmlNodeQuadSet(VrmlScene *);

  virtual Viewer::Object insertGeometry(Viewer *v);

  virtual VrmlNodeQuadSet *toQuadSet() const;
};
}
#endif // _VRMLNODEQUADSET_
