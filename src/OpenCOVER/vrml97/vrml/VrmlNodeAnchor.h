/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeAnchor.h

#ifndef _VRMLNODEANCHOR_
#define _VRMLNODEANCHOR_

#include "VrmlNodeGroup.h"
#include "VrmlMFString.h"
#include "VrmlSFString.h"

namespace vrml
{

class VrmlNodeType;
class VrmlScene;

class VRMLEXPORT VrmlNodeAnchor : public VrmlNodeGroup
{

public:
    // Define the built in VrmlNodeType:: "Anchor"
    static void initFields(VrmlNodeAnchor *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeAnchor(VrmlScene *);
    VrmlNodeAnchor(const VrmlNodeAnchor &);

    virtual VrmlNodeAnchor *toAnchor() const override;

    virtual void render(Viewer *) override;

    void activate();

    const char *description()
    {
        return d_description.get();
    }
    const char *url()
    {
        return d_url.size() > 0 ? d_url[0] : 0;
    }

    bool isOnlyGeometry() const override;

protected:
    VrmlSFString d_description;
    VrmlMFString d_parameter;
    VrmlMFString d_url;
};
}
#endif // _VRMLNODEANCHOR_
