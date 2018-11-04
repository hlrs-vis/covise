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
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const override;

    VrmlNodeAnchor(VrmlScene *);
    VrmlNodeAnchor(const VrmlNodeAnchor &);
    virtual ~VrmlNodeAnchor();

    // Copy the node.
    virtual VrmlNode *cloneMe() const override;

    virtual VrmlNodeAnchor *toAnchor() const override;

    virtual std::ostream &printFields(std::ostream &os, int indent) override;

    virtual void render(Viewer *) override;

    void activate();

    virtual void setField(const char *fieldName,
                          const VrmlField &fieldValue) override;
    const VrmlField *getField(const char *fieldName) const override;

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
