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

#include "VrmlNode.h"
#include "VrmlSFNode.h"

#include "Viewer.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeShape : public VrmlNodeChild
{

public:
    // Define the fields of Shape nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeShape(VrmlScene *);
    virtual ~VrmlNodeShape();

    virtual VrmlNode *cloneMe() const;
    virtual void cloneChildren(VrmlNamespace *);

    virtual bool isModified() const;

    virtual void clearFlags();

    virtual VrmlNodeShape *toShape() const;

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void copyRoutes(VrmlNamespace *ns);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void render(Viewer *);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

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

private:
    VrmlSFNode d_appearance;
    VrmlSFNode d_geometry;
    VrmlSFNode d_effect;

    Viewer::Object d_viewerObject; // move to VrmlNode.h ? ...
};
}
#endif //_VRMLNODESHAPE_
