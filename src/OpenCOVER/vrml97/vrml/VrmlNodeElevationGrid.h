/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//  %W% %G%
//  VrmlNodeElevationGrid.h

#ifndef _VRMLNODEELEVATIONGRID_
#define _VRMLNODEELEVATIONGRID_

#include "VrmlNodeGeometry.h"

#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFInt.h"
#include "VrmlSFNode.h"

#include "VrmlMFFloat.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeElevationGrid : public VrmlNodeGeometry
{

public:
    // Define the fields of elevationGrid nodes
    static void initFields(VrmlNodeElevationGrid *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeElevationGrid(VrmlScene *);

    virtual void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;
    virtual void clearFlags();

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual Viewer::Object insertGeometry(Viewer *);

    virtual VrmlNodeColor *color();

    //LarryD Mar 09/99
    virtual VrmlNodeElevationGrid *toElevationGrid() const;

    virtual VrmlNode *getNormal(); //LarryD Mar 09/99
    virtual VrmlNode *getTexCoord(); //LarryD Mar 09/99
    virtual bool getCcw() //LarryD Mar 09/99
    {
        return d_ccw.get();
    }
    virtual bool getColorPerVertex() //LarryD Mar 09/99
    {
        return d_colorPerVertex.get();
    }
    virtual float getCreaseAngle() //LarryD Mar 09/99
    {
        return d_creaseAngle.get();
    }
    //LarryD Mar 09/99
    virtual const VrmlMFFloat &getHeight() const;
    virtual bool getNormalPerVertex() //LarryD Mar 09/99
    {
        return d_normalPerVertex.get();
    }
    virtual bool getSolid() //LarryD Mar 09/99
    {
        return d_solid.get();
    }
    virtual int getXDimension() //LarryD Mar 09/99
    {
        return d_xDimension.get();
    }
    virtual float getXSpacing() //LarryD Mar 09/99
    {
        return d_xSpacing.get();
    }
    virtual int getZDimension() //LarryD Mar 09/99
    {
        return d_zDimension.get();
    }
    virtual float getZSpacing() //LarryD Mar 09/99
    {
        return d_zSpacing.get();
    }

protected:
    VrmlSFNode d_color;
    VrmlSFNode d_normal;
    VrmlSFNode d_texCoord;

    VrmlSFBool d_ccw;
    VrmlSFBool d_colorPerVertex;
    VrmlSFFloat d_creaseAngle;
    VrmlMFFloat d_height;
    VrmlSFBool d_normalPerVertex;
    VrmlSFBool d_solid;
    VrmlSFInt d_xDimension;
    VrmlSFFloat d_xSpacing;
    VrmlSFInt d_zDimension;
    VrmlSFFloat d_zSpacing;
};
}
#endif //_VRMLNODEELEVATIONGRID_
