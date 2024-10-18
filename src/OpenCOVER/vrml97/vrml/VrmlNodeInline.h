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

    static void initFields(VrmlNodeInline *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeInline(VrmlScene *);
    virtual ~VrmlNodeInline();

    virtual VrmlNodeInline *toInline() const;

    virtual void addToScene(VrmlScene *s, const char *relativeUrl);

    virtual void render(Viewer *viewer);

    void load(const char *relativeUrl, int parentId = -1);

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
