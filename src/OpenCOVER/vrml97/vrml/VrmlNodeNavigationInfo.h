/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeNavigationInfo.h

#ifndef _VRMLNODENAVIGATIONINFO_
#define _VRMLNODENAVIGATIONINFO_

#include "VrmlNode.h"
#include "VrmlMFFloat.h"
#include "VrmlMFString.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"

#include "VrmlNodeChild.h"

namespace vrml
{

class VRMLEXPORT VrmlNodeNavigationInfo : public VrmlNodeChild
{

public:
    bool lastBind;

    // Define the built in VrmlNodeType:: "NavigationInfo"
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeNavigationInfo(VrmlScene *scene);
    virtual ~VrmlNodeNavigationInfo();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeNavigationInfo *toNavigationInfo() const;

    // Bindable nodes must notify the scene of their existence.
    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    float *avatarSize()
    {
        return d_avatarSize.get();
    }
    float scale()
    {
        return d_scale.get();
    }
    float getNear()
    {
        return d_near.get();
    }
    float getFar()
    {
        return d_far.get();
    }
    float lastScale()
    {
        return d_lastScale.get();
    }
    bool headlightOn()
    {
        return d_headlight.get();
    }
    float speed()
    {
        return d_speed.get();
    }
    float visibilityLimit()
    {
        return d_visibilityLimit.get();
    }
    char **navTypes()
    {
        return d_type.get();
    }
    char **transitionTypes()
    {
        return d_transitionType.get();
    }

private:
    VrmlMFFloat d_avatarSize;
    VrmlSFBool d_headlight;
    VrmlSFFloat d_scale;
    VrmlSFFloat d_lastScale;
    VrmlSFFloat d_speed;
    VrmlSFFloat d_near;
    VrmlSFFloat d_far;
    VrmlMFString d_type;
    VrmlMFString d_transitionType;
    VrmlMFFloat d_transitionTime;
    VrmlSFFloat d_visibilityLimit;
};
}
#endif //_VRMLNODENAVIGATIONINFO_
