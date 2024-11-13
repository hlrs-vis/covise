/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeGeometry.h

#ifndef _VRMLNODEGEOMETRY_
#define _VRMLNODEGEOMETRY_

#include "VrmlNode.h"
#include "Viewer.h"
#include <string>
namespace vrml
{

class VRMLEXPORT VrmlNodeColor;

class VRMLEXPORT VrmlNodeGeometry : public VrmlNode
{

public:
    // Define the fields of all built in geometry nodes
    static void initFields(VrmlNodeGeometry *node, VrmlNodeType *t);

    VrmlNodeGeometry(VrmlScene *, const std::string &name);

    // Geometry nodes need only define insertGeometry(), not render().
    virtual void render(Viewer *) override;

    virtual Viewer::Object insertGeometry(Viewer *) = 0;

    virtual VrmlNodeColor *color();
    virtual Viewer::Object getViewerObject()
    {
        return d_viewerObject;
    };

    virtual bool isOnlyGeometry() const override;

protected:
    Viewer::Object d_viewerObject; // move to VrmlNode? ...
};
}
#endif // _VRMLNODEGEOMETRY_
