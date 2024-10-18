/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeBackground.h

#ifndef _VRMLNODEBACKGROUND_
#define _VRMLNODEBACKGROUND_

#include "VrmlNode.h"
#include "VrmlField.h"

#include "Image.h"
#include "VrmlMFColor.h"
#include "VrmlMFFloat.h"
#include "VrmlMFString.h"
#include "VrmlSFString.h"
#include "Viewer.h"

#include "VrmlNodeChild.h"

namespace vrml
{
//class Viewer;
class VrmlScene;

class VRMLEXPORT VrmlNodeBackground : public VrmlNodeChild
{

public:
    // Define the fields of Background nodes

    static void initFields(VrmlNodeBackground *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeBackground(VrmlScene *);
    virtual ~VrmlNodeBackground();

    virtual VrmlNodeBackground *toBackground() const;

    virtual void addToScene(VrmlScene *s, const char *relativeUrl);

    // render backgrounds once per scene, not via the render() method
    void renderBindable(Viewer *);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);


    int nGroundAngles()
    {
        return d_groundAngle.size();
    }
    float *groundAngle()
    {
        return d_groundAngle.get();
    }
    float *groundColor()
    {
        return d_groundColor.get();
    }

    int nSkyAngles()
    {
        return d_skyAngle.size();
    }
    float *skyAngle()
    {
        return d_skyAngle.get();
    }
    float *skyColor()
    {
        return d_skyColor.get();
    }

private:
    VrmlMFFloat d_groundAngle;
    VrmlMFColor d_groundColor;

    VrmlMFString d_backUrl;
    VrmlMFString d_bottomUrl;
    VrmlMFString d_frontUrl;
    VrmlMFString d_leftUrl;
    VrmlMFString d_rightUrl;
    VrmlMFString d_topUrl;

    VrmlMFFloat d_skyAngle;
    VrmlMFColor d_skyColor;

    VrmlSFString d_relativeUrl;

    // Texture caches
    Image *d_texPtr[6];
    Image d_tex[6];

    // Display list object for background
    Viewer::Object d_viewerObject;
};
}
#endif //_VRMLNODEBACKGROUND_
