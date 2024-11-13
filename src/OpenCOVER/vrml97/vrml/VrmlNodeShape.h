/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeShape.h

#ifndef _VRMLNODESHAPE_
#define _VRMLNODESHAPE_

#include "VrmlNodeChild.h"
#include "VrmlSFNode.h"

#include "Viewer.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeShape : public VrmlNodeChild
{

public:
    // Define the fields of Shape nodes
    static void initFields(VrmlNodeShape *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeShape(VrmlScene *);

    virtual void cloneChildren(VrmlNamespace *) override;

    virtual bool isModified() const override;

    virtual void clearFlags() override;

    virtual void addToScene(VrmlScene *s, const char *relUrl) override;

    virtual void copyRoutes(VrmlNamespace *ns) override;

    virtual void render(Viewer *) override;

    virtual VrmlNode *getAppearance()
    {
        return d_appearance.get();
    }
    virtual VrmlNode *getGeometry()
    {
        return d_geometry.get();
    }
    virtual VrmlNode *getEffect()
    {
        return d_effect.get();
    }
    virtual Viewer::Object getViewerObject()
    {
        return d_viewerObject;
    };

    bool isOnlyGeometry() const override;

private:
    VrmlSFNode d_appearance;
    VrmlSFNode d_geometry;
    VrmlSFNode d_effect;

    Viewer::Object d_viewerObject; // move to VrmlNode.h ? ...
};
}
#endif //_VRMLNODESHAPE_
