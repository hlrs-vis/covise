/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeAudioClip.h
//    contributed by Kumaran Santhanam

#ifndef _VRMLNODEAUDIOCLIP_
#define _VRMLNODEAUDIOCLIP_

#include "VrmlNode.h"
#include "VrmlMFString.h"
#include "VrmlSFBool.h"
#include "VrmlSFFloat.h"
#include "VrmlSFString.h"
#include "VrmlSFTime.h"
#include "VrmlSFVec3f.h"

namespace vrml
{

class Audio;
class Doc;

class VRMLEXPORT VrmlNodeAudioClip : public VrmlNode
{

public:
    // Define the fields of AudioClip nodes
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeAudioClip(VrmlScene *);
    VrmlNodeAudioClip(const VrmlNodeAudioClip &);
    virtual ~VrmlNodeAudioClip();

    // Copy the node.
    virtual VrmlNode *cloneMe() const;

    virtual void addToScene(VrmlScene *s, const char *relativeUrl);

    // an update. Renderable nodes need to redefine this.
    virtual void update(VrmlSFTime &now);

    virtual VrmlNodeAudioClip *toAudioClip() const;

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);
    const VrmlField *getField(const char *fieldName) const;

    const Audio *getAudio() const;

    bool isAudible(VrmlSFTime &now) const;

    double currentCliptime(VrmlSFTime &now) const;

private:
    VrmlSFString d_description;
    VrmlSFBool d_loop;
    VrmlSFFloat d_pitch;
    VrmlSFTime d_startTime;
    VrmlSFTime d_stopTime;
    VrmlMFString d_url;

    VrmlSFString d_relativeUrl;

    VrmlSFTime d_duration;
    VrmlSFBool d_isActive;

    Audio *d_audio;
    bool d_url_modified;
    Doc *_doc;
    double lastTime;
    bool lastActive;
};
}
#endif //_VRMLNODEAUDIOCLIP_
