/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCOVER.h

#ifndef _VrmlNodeCOVER_
#define _VrmlNodeCOVER_

#include "VrmlNode.h"
#include "VrmlSFString.h"
#include "VrmlSFInt.h"
#include "VrmlSFVec3f.h"
#include "VrmlSFRotation.h"
#include "VrmlNodeChild.h"

#ifdef VRML_PUI
#include <vrui/coPocketUI.h>
#endif

namespace vrml
{

class VRMLEXPORT VrmlNodeCOVER
    : public VrmlNodeChild
#ifdef VRML_PUI
      ,
      public coPUIListener
#endif
{

public:
    enum KeyEventType
    {
        Unknown,
        Press,
        Release
    };

    // Define the built in VrmlNodeType:: "COVER"
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    VrmlNodeCOVER(VrmlScene *scene);
    virtual ~VrmlNodeCOVER();

    virtual VrmlNode *cloneMe() const;

    virtual VrmlNodeCOVER *toCOVER() const;

    virtual void update(double timeStamp);

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual std::ostream &printFields(std::ostream &os, int indent);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);

    // process key events
    void keyEvent(enum KeyEventType type, const char *keyModString);

    // process remote key events, called by eventQueue
    void remoteKeyEvent(enum KeyEventType type, const char *keyModString);

#ifdef VRML_PUI
    virtual void pocketPressEvent(coPUIElement *pUIItem);

    virtual void pocketEvent(coPUIElement *pUIItem);
#endif

    double transformations[15][16];

private:
    VrmlSFString d_keyPressed;
    VrmlSFString d_keyReleased;
    VrmlSFString d_localKeyPressed;
    VrmlSFString d_localKeyReleased;
    VrmlSFInt d_soundEnvironment;
    VrmlSFInt d_animationTimeStep;
    VrmlSFInt d_activePerson;
    VrmlSFVec3f d_avatar1Position;
    VrmlSFRotation d_avatar1Orientation;
    VrmlSFVec3f d_localPosition;
    VrmlSFRotation d_localOrientation;
    VrmlSFVec3f d_localViewerPosition;
    VrmlSFRotation d_localViewerOrientation;
    VrmlSFVec3f d_position1;
    VrmlSFVec3f d_position2;
    VrmlSFVec3f d_position3;
    VrmlSFVec3f d_position4;
    VrmlSFVec3f d_position5;
    VrmlSFVec3f d_position6;
    VrmlSFVec3f d_position7;
    VrmlSFVec3f d_position8;
    VrmlSFVec3f d_position9;
    VrmlSFVec3f d_position10;
    VrmlSFVec3f d_position11;
    VrmlSFVec3f d_position12;
    VrmlSFVec3f d_position13;
    VrmlSFVec3f d_position14;
    VrmlSFVec3f d_position15;
    VrmlSFRotation d_orientation1;
    VrmlSFRotation d_orientation2;
    VrmlSFRotation d_orientation3;
    VrmlSFRotation d_orientation4;
    VrmlSFRotation d_orientation5;
    VrmlSFRotation d_orientation6;
    VrmlSFRotation d_orientation7;
    VrmlSFRotation d_orientation8;
    VrmlSFRotation d_orientation9;
    VrmlSFRotation d_orientation10;
    VrmlSFRotation d_orientation11;
    VrmlSFRotation d_orientation12;
    VrmlSFRotation d_orientation13;
    VrmlSFRotation d_orientation14;
    VrmlSFRotation d_orientation15;
    VrmlSFString d_saveTimestamp;
	VrmlSFString d_loadPlugin;
#ifdef VRML_PUI
    coPUITab *pTab1;
    coPUIEditField *pText;
    coPUIBitmapButton *flyButton;
    coPUIBitmapButton *driveButton;
    coPUIBitmapButton *walkButton;
    coPUIBitmapButton *xformButton;
    coPUIFKeys *fKeys;
#endif
};

extern VRMLEXPORT VrmlNodeCOVER *theCOVER;
}
#endif //_VrmlNodeCOVER_
