/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeMovieTexture.h

#ifndef _VRMLNODEMOVIETEXTURE_
#define _VRMLNODEMOVIETEXTURE_

#include "VrmlNodeTexture.h"
#include "VrmlMFString.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFTime.h"
#include "VrmlSFInt.h"
#include "Viewer.h"

namespace vrml
{

class Image;

class VRMLEXPORT VrmlNodeMovieTexture : public VrmlNodeTexture
{

public:
    // Define the fields of MovieTexture nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const override;

    VrmlNodeMovieTexture(VrmlScene *);
    virtual ~VrmlNodeMovieTexture();

    virtual VrmlNode *cloneMe() const override;

    virtual VrmlNodeMovieTexture *toMovieTexture() const override;

    virtual void addToScene(VrmlScene *s, const char *relUrl) override;

    virtual std::ostream &printFields(std::ostream &os, int indent) override;

    void update(VrmlSFTime &now);

    virtual void render(Viewer *) override;

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue) override;

    virtual const VrmlField *getField(const char *fieldName) const override;
    virtual void setField(const char *fieldName, const VrmlField &fieldValue) override;

    virtual int nComponents() override;
    virtual int width() override;
    virtual int height() override;
    virtual int nFrames() override;
    virtual void stopped();
    virtual unsigned char *pixels() override;

    bool isOnlyGeometry() const override;
private:
    VrmlSFBool d_loop;
    VrmlSFFloat d_speed;
    VrmlSFTime d_startTime;
    VrmlSFTime d_stopTime;

    VrmlMFString d_url;
    VrmlSFBool d_repeatS;
    VrmlSFBool d_repeatT;

    VrmlSFTime d_duration;
    VrmlSFBool d_isActive;
    VrmlSFBool d_environment;
    VrmlSFInt d_anisotropy;
    VrmlSFInt d_filter;

    Image *d_image;
    bool imageChanged;
    int d_frame, d_lastFrame;
    float mpegStaticLength;
    double d_lastFrameTime;
    movieProperties movProp;

    Viewer::TextureObject d_texObject;
};
}
#endif // _VRMLNODEMOVIETEXTURE_
