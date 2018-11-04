/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeInline.h

#ifndef _VRMLNODEINLINE_
#define _VRMLNODEINLINE_

#include "VrmlNodeGroup.h"
#include "VrmlMFString.h"
#include "Viewer.h"

namespace vrml
{

class VrmlNamespace;
class VrmlNodeType;
class VrmlScene;

class VRMLEXPORT VrmlNodeInline : public VrmlNodeGroup
{

public:
    // Define the built in VrmlNodeType:: "Inline"
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeInline(VrmlScene *);
    virtual ~VrmlNodeInline();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeInline *toInline() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void addToScene(VrmlScene *s, const char *relativeUrl);

    virtual void render(Viewer *viewer);

    virtual void setField(const char *fieldName,
                          const VrmlField &fieldValue);
    virtual const VrmlField *getField(const char *fieldName) const;

    void load(const char *relativeUrl);

    virtual VrmlNode *findInside(const char *exportName);

protected:
    VrmlMFString d_url;

    VrmlNamespace *d_namespace;
    Viewer::Object sgObject;

    bool d_hasLoaded;
    bool d_wasCached = false;
};
}
#endif // _VRMLNODEINLINE_
