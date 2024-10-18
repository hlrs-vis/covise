/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeFog.h

#ifndef _VRMLNODEFOG_
#define _VRMLNODEFOG_

#include "VrmlNode.h"

#include "VrmlSFColor.h"
#include "VrmlSFFloat.h"
#include "VrmlSFString.h"
#include "VrmlNodeChild.h"

namespace vrml
{

class VrmlScene;

class VRMLEXPORT VrmlNodeFog : public VrmlNodeChild
{

public:

    static void initFields(VrmlNodeFog *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeFog(VrmlScene *);
    virtual ~VrmlNodeFog();

    virtual VrmlNodeFog *toFog() const;

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    float *color()
    {
        return d_color.get();
    }
    const char *fogType()
    {
        return d_fogType.get();
    }
    float visibilityRange()
    {
        return d_visibilityRange.get();
    }

private:
    VrmlSFColor d_color;
    VrmlSFString d_fogType;
    VrmlSFFloat d_visibilityRange;
};
}
#endif //_VRMLNODEFOG_
