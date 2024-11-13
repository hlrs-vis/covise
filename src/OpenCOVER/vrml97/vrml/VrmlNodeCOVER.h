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

#include <array>

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
    static void initFields(VrmlNodeCOVER *node, VrmlNodeType *t);
    static const char *name();

    VrmlNodeCOVER(VrmlScene *scene);
    virtual ~VrmlNodeCOVER();

    virtual void update(double timeStamp);

    virtual void addToScene(VrmlScene *s, const char *relUrl);


    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);


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
    
    static const int NUM_POSITIONS = 15;
    std::array<VrmlSFVec3f, NUM_POSITIONS> d_positions;
    std::array<VrmlSFRotation, NUM_POSITIONS> d_orientations;
   
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
