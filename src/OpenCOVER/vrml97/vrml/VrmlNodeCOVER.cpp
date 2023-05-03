/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  VrmlNodeCOVER.cpp

#include "VrmlNodeCOVER.h"
#include "MathUtils.h"
#include "VrmlNodeType.h"
#include "VrmlScene.h"
#include "coEventQueue.h"
#define XK_MISCELLANY
#if !defined(_WIN32) && !defined(__APPLE__)
#include <X11/X.h>
#include <X11/keysymdef.h>
#else
#define PFUDEV_NULL 0

/*
 * Special performer device mappings
 * Function keys MUST be numbers 1-12
 */
#define PFUDEV_F1KEY 1
#define PFUDEV_F2KEY 2
#define PFUDEV_F3KEY 3
#define PFUDEV_F4KEY 4
#define PFUDEV_F5KEY 5
#define PFUDEV_F6KEY 6
#define PFUDEV_F7KEY 7
#define PFUDEV_F8KEY 8
#define PFUDEV_F9KEY 9
#define PFUDEV_F10KEY 10
#define PFUDEV_F11KEY 11
#define PFUDEV_F12KEY 12
#define PFUDEV_KEYBD 13
#define PFUDEV_REDRAW 14
#define PFUDEV_WINQUIT 15
/*
 * ESC is offered as a convenience for when using GL events.
 * It generally should be acquired through PFUDEV_KEYBD (== 27)
 */
#define PFUDEV_ESCKEY 16
#define PFUDEV_LEFTARROWKEY 17
#define PFUDEV_DOWNARROWKEY 18
#define PFUDEV_RIGHTARROWKEY 19
#define PFUDEV_UPARROWKEY 20

#define PFUDEV_PRINTSCREENKEY 21

#define PFUDEV_MAX 22

#define PFUDEV_MAX_DEVS 256

/* Masks for 'flags' in pfuMouse */
#define PFUDEV_MOUSE_RIGHT_DOWN 0x0001
#define PFUDEV_MOUSE_MIDDLE_DOWN 0x0002
#define PFUDEV_MOUSE_LEFT_DOWN 0x0004
#define PFUDEV_MOUSE_DOWN_MASK 0x0007

/* Special masks for pfuEventStream 'buttonFlags' and pfuMouse flags */
#define PFUDEV_MOD_SHIFT 0x0010
#define PFUDEV_MOD_LEFT_SHIFT 0x0020
#define PFUDEV_MOD_LEFT_SHIFT_SET (PFUDEV_MOD_LEFT_SHIFT | PFUDEV_MOD_SHIFT)
#define PFUDEV_MOD_RIGHT_SHIFT 0x0040
#define PFUDEV_MOD_RIGHT_SHIFT_SET (PFUDEV_MOD_RIGHT_SHIFT | PFUDEV_MOD_SHIFT)
#define PFUDEV_MOD_SHIFT_MASK 0x0070
#define PFUDEV_MOD_CTRL 0x0080
#define PFUDEV_MOD_LEFT_CTRL 0x0100
#define PFUDEV_MOD_LEFT_CTRL_SET (PFUDEV_MOD_LEFT_CTRL | PFUDEV_MOD_CTRL)
#define PFUDEV_MOD_RIGHT_CTRL 0x0200
#define PFUDEV_MOD_RIGHT_CTRL_SET (PFUDEV_MOD_RIGHT_CTRL | PFUDEV_MOD_CTRL)
#define PFUDEV_MOD_CAPS_LOCK (0x0400 | PFUDEV_MOD_CTRL)
#define PFUDEV_MOD_CTRL_MASK 0x0780
#define PFUDEV_MOD_ALT 0x0800
#define PFUDEV_MOD_LEFT_ALT 0x1000
#define PFUDEV_MOD_LEFT_ALT_SET (PFUDEV_MOD_LEFT_ALT | PFUDEV_MOD_ALT)
#define PFUDEV_MOD_RIGHT_ALT 0x2000
#define PFUDEV_MOD_RIGHT_ALT_SET (PFUDEV_MOD_RIGHT_ALT | PFUDEV_MOD_ALT)
#define PFUDEV_MOD_ALT_MASK 0x3800
#define PFUDEV_MOD_MASK 0x3ff0
#endif
#include "Player.h"

#if defined(_WIN32) || defined(__APPLE__)
#define KeyPress 1
#define KeyRelease 2
#endif

using std::cerr;
using std::endl;

namespace vrml
{
VrmlNodeCOVER *theCOVER = NULL;
}

using namespace vrml;

//  COVER factory.
//  Since NavInfo is a bindable child node, the first one created needs
//  to notify its containing scene.

static VrmlNode *creator(VrmlScene *scene)
{
    if (theCOVER == NULL)
    {
        theCOVER = new VrmlNodeCOVER(scene);
    }
    return theCOVER;
}

// Define the built in VrmlNodeType:: "COVER" fields

VrmlNodeType *VrmlNodeCOVER::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("COVER", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("position1", VrmlField::SFVEC3F);
    t->addExposedField("position2", VrmlField::SFVEC3F);
    t->addExposedField("position3", VrmlField::SFVEC3F);
    t->addExposedField("position4", VrmlField::SFVEC3F);
    t->addExposedField("position5", VrmlField::SFVEC3F);
    t->addExposedField("position6", VrmlField::SFVEC3F);
    t->addExposedField("position7", VrmlField::SFVEC3F);
    t->addExposedField("position8", VrmlField::SFVEC3F);
    t->addExposedField("position9", VrmlField::SFVEC3F);
    t->addExposedField("position10", VrmlField::SFVEC3F);
    t->addExposedField("position11", VrmlField::SFVEC3F);
    t->addExposedField("position12", VrmlField::SFVEC3F);
    t->addExposedField("position13", VrmlField::SFVEC3F);
    t->addExposedField("position14", VrmlField::SFVEC3F);
    t->addExposedField("position15", VrmlField::SFVEC3F);
    t->addExposedField("orientation1", VrmlField::SFROTATION);
    t->addExposedField("orientation2", VrmlField::SFROTATION);
    t->addExposedField("orientation3", VrmlField::SFROTATION);
    t->addExposedField("orientation4", VrmlField::SFROTATION);
    t->addExposedField("orientation5", VrmlField::SFROTATION);
    t->addExposedField("orientation6", VrmlField::SFROTATION);
    t->addExposedField("orientation7", VrmlField::SFROTATION);
    t->addExposedField("orientation8", VrmlField::SFROTATION);
    t->addExposedField("orientation9", VrmlField::SFROTATION);
    t->addExposedField("orientation10", VrmlField::SFROTATION);
    t->addExposedField("orientation11", VrmlField::SFROTATION);
    t->addExposedField("orientation12", VrmlField::SFROTATION);
    t->addExposedField("orientation13", VrmlField::SFROTATION);
    t->addExposedField("orientation14", VrmlField::SFROTATION);
    t->addExposedField("orientation15", VrmlField::SFROTATION);
    //t->addEventIn("avatarSize", VrmlField::MFFLOAT);
    t->addEventOut("localKeyPressed", VrmlField::SFSTRING);
    t->addEventOut("localKeyReleased", VrmlField::SFSTRING);
    t->addEventOut("keyPressed", VrmlField::SFSTRING);
    t->addEventOut("keyReleased", VrmlField::SFSTRING);
    t->addEventOut("avatar1Position", VrmlField::SFVEC3F);
    t->addEventOut("avatar1Orientation", VrmlField::SFROTATION);
    t->addEventOut("localPosition", VrmlField::SFVEC3F);
    t->addEventOut("localOrientation", VrmlField::SFROTATION);
    t->addEventOut("localViewerPosition", VrmlField::SFVEC3F);
    t->addEventOut("localViewerOrientation", VrmlField::SFROTATION);
    t->addExposedField("soundEnvironment", VrmlField::SFINT32);
    t->addExposedField("animationTimeStep", VrmlField::SFINT32);
    t->addExposedField("activePerson", VrmlField::SFINT32);
    t->addEventIn("saveTimestamp", VrmlField::SFSTRING);
	t->addExposedField("loadPlugin", VrmlField::SFSTRING);

    return t;
}

VrmlNodeType *VrmlNodeCOVER::nodeType() const { return defineType(0); }

VrmlNodeCOVER::VrmlNodeCOVER(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    d_soundEnvironment.set(26);
    d_animationTimeStep.set(0);
    d_activePerson.set(0);
    d_saveTimestamp.set("");
	d_loadPlugin.set("");
    reference();
#ifdef VRML_PUI
    pTab1 = new coPUITab("VRML Keyboard");
    pText = new coPUIEditField("test", pTab1->getID());
    pText->setPos(10, 10);
    pText->setImmediate(true);
    pText->setEventListener(this);
    flyButton = new coPUIBitmapButton("Fly.bmp", pTab1->getID());
    flyButton->setPos(10, 50);
    flyButton->setEventListener(this);
    driveButton = new coPUIBitmapButton("Drive.bmp", pTab1->getID());
    driveButton->setPos(60, 50);
    driveButton->setEventListener(this);
    walkButton = new coPUIBitmapButton("Walk.bmp", pTab1->getID());
    walkButton->setPos(10, 100);
    walkButton->setEventListener(this);
    xformButton = new coPUIBitmapButton("XForm.bmp", pTab1->getID());
    xformButton->setPos(60, 100);
    xformButton->setEventListener(this);
    fKeys = new coPUIFKeys("fKeys", 0);
    fKeys->setEventListener(this);
#endif
    d_position1.set(1000000, 1000000, 1000000);
    d_position2.set(1000000, 1000000, 1000000);
    d_position3.set(1000000, 1000000, 1000000);
    d_position4.set(1000000, 1000000, 1000000);
    d_position5.set(1000000, 1000000, 1000000);
    d_position6.set(1000000, 1000000, 1000000);
    d_position7.set(1000000, 1000000, 1000000);
    d_position8.set(1000000, 1000000, 1000000);
    d_position9.set(1000000, 1000000, 1000000);
    d_position10.set(1000000, 1000000, 1000000);
    d_position11.set(1000000, 1000000, 1000000);
    d_position12.set(1000000, 1000000, 1000000);
    d_position13.set(1000000, 1000000, 1000000);
    d_position14.set(1000000, 1000000, 1000000);
    d_position15.set(1000000, 1000000, 1000000);
    d_orientation1.set(0, 1, 0, 0);
    d_orientation2.set(0, 1, 0, 0);
    d_orientation3.set(0, 1, 0, 0);
    d_orientation4.set(0, 1, 0, 0);
    d_orientation5.set(0, 1, 0, 0);
    d_orientation6.set(0, 1, 0, 0);
    d_orientation7.set(0, 1, 0, 0);
    d_orientation8.set(0, 1, 0, 0);
    d_orientation9.set(0, 1, 0, 0);
    d_orientation10.set(0, 1, 0, 0);
    d_orientation11.set(0, 1, 0, 0);
    d_orientation12.set(0, 1, 0, 0);
    d_orientation13.set(0, 1, 0, 0);
    d_orientation14.set(0, 1, 0, 0);
    d_orientation15.set(0, 1, 0, 0);
}

#ifdef VRML_PUI
void VrmlNodeCOVER::pocketPressEvent(coPUIElement * /*pUIItem*/)
{
    cerr << "button Event" << endl;
}

void VrmlNodeCOVER::pocketEvent(coPUIElement *pUIItem)
{
    cerr << "bpocket Event" << endl;
    if (pUIItem == fKeys)
    {
        if (strlen(fKeys->FKey) > 0)
        {
            double timeStamp = System::the->time();
            d_keyPressed.set(fKeys->FKey);
            eventOut(timeStamp, "keyPressed", d_keyPressed);
            d_localKeyPressed.set(fKeys->FKey);
            eventOut(timeStamp, "localKeyPressed", d_keyPressed);
            d_scene->getIncomingSensorEventQueue()->sendKeyEvent(KeyPress, fKeys->FKey);
        }
    }
    if (pUIItem == pText)
    {
        const char *text = pText->getText();
        if (strlen(text) > 0)
        {
            char keystringMod[200];

            keystringMod[0] = text[strlen(text) - 1];
            keystringMod[1] = '\0';

            /* keystringMod[0]='\0';
          if(mod&MOD_CTRL)
          {
              strcat(keystringMod,"Ctrl-");
          }
          if(mod&MOD_ALT)
          {
              strcat(keystringMod,"Alt-");
          }
          if(mod&MOD_ALT_GR)
          {
         strcat(keystringMod,"AltGr-");
         }
         strcat(keystringMod,keystring);
         */
            double timeStamp = System::the->time();
            d_keyPressed.set(keystringMod);
            eventOut(timeStamp, "keyPressed", d_keyPressed);
            d_localKeyPressed.set(keystringMod);
            eventOut(timeStamp, "localKeyPressed", d_keyPressed);
            d_scene->getIncomingSensorEventQueue()->sendKeyEvent(KeyPress, keystringMod);
            pText->setText("");
        }
    }
}
#endif

VrmlNodeCOVER::~VrmlNodeCOVER()
{
    cerr << "This node (COVER) should never be deleted!!!!\n";
}

VrmlNode *VrmlNodeCOVER::cloneMe() const
{
    return new VrmlNodeCOVER(*this);
}

void VrmlNodeCOVER::update(double timeNow)
{
    double tmpRot[16], tmpTrans[16];

    Mrotation(tmpRot, d_orientation1.get(), d_orientation1.r());
    Mtrans(tmpTrans, d_position1.get());
    Mmult(transformations[0], tmpRot, tmpTrans);
    //for(int u=0;u<4;u++)
    //cerr << "vrml:" << transformations[0][u][0] << " "transformations[0][u][1] << " "transformations[0][u][2] << " "transformations[0][u][3] << " " << endl;
    //cerr << transformations[0][u][0] << " "transformations[0][u][1] << " "transformations[0][u][2] << " "transformations[0][u][3] << " " << endl;
    //cerr << transformations[0][u][0] << " "transformations[0][u][1] << " "transformations[0][u][2] << " "transformations[0][u][3] << " " << endl;

    Mrotation(tmpRot, d_orientation2.get(), d_orientation2.r());
    Mtrans(tmpTrans, d_position2.get());
    Mmult(transformations[1], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation3.get(), d_orientation3.r());
    Mtrans(tmpTrans, d_position3.get());
    Mmult(transformations[2], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation4.get(), d_orientation4.r());
    Mtrans(tmpTrans, d_position4.get());
    Mmult(transformations[3], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation5.get(), d_orientation5.r());
    Mtrans(tmpTrans, d_position5.get());
    Mmult(transformations[4], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation6.get(), d_orientation6.r());
    Mtrans(tmpTrans, d_position6.get());
    Mmult(transformations[5], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation7.get(), d_orientation7.r());
    Mtrans(tmpTrans, d_position7.get());
    Mmult(transformations[6], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation8.get(), d_orientation8.r());
    Mtrans(tmpTrans, d_position8.get());
    Mmult(transformations[7], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation9.get(), d_orientation9.r());
    Mtrans(tmpTrans, d_position9.get());
    Mmult(transformations[8], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation10.get(), d_orientation10.r());
    Mtrans(tmpTrans, d_position10.get());
    Mmult(transformations[9], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation11.get(), d_orientation11.r());
    Mtrans(tmpTrans, d_position11.get());
    Mmult(transformations[10], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation12.get(), d_orientation12.r());
    Mtrans(tmpTrans, d_position12.get());
    Mmult(transformations[11], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation13.get(), d_orientation13.r());
    Mtrans(tmpTrans, d_position13.get());
    Mmult(transformations[12], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation14.get(), d_orientation14.r());
    Mtrans(tmpTrans, d_position14.get());
    Mmult(transformations[13], tmpRot, tmpTrans);

    Mrotation(tmpRot, d_orientation15.get(), d_orientation15.r());
    Mtrans(tmpTrans, d_position15.get());
    Mmult(transformations[14], tmpRot, tmpTrans);

    float pos[3];
    float ori[4];
    System::the->getAvatarPositionAndOrientation(0, pos, ori);
    d_avatar1Position.set(pos[0], pos[1], pos[2]);
    d_avatar1Orientation.set(ori[0], ori[1], ori[2], ori[3]);
    eventOut(timeNow, "avatar1Position", d_avatar1Position);
    eventOut(timeNow, "avatar1Orientation", d_avatar1Orientation);

    System::the->getViewerFeetPositionAndOrientation(pos, ori);
    d_localPosition.set(pos[0], pos[1], pos[2]);
    d_localOrientation.set(ori[0], ori[1], ori[2], ori[3]);
    eventOut(timeNow, "localPosition", d_localPosition);
    eventOut(timeNow, "localOrientation", d_localOrientation);

    System::the->getLocalViewerPositionAndOrientation(pos, ori);
    d_localViewerPosition.set(pos[0], pos[1], pos[2]);
    d_localViewerOrientation.set(ori[0], ori[1], ori[2], ori[3]);
    eventOut(timeNow, "localViewerPosition", d_localViewerPosition);
    eventOut(timeNow, "localViewerOrientation", d_localViewerOrientation);
}

VrmlNodeCOVER *VrmlNodeCOVER::toCOVER() const
{
    return (VrmlNodeCOVER *)this;
}

void VrmlNodeCOVER::addToScene(VrmlScene *s, const char *)
{
    d_scene = s;
}

std::ostream &VrmlNodeCOVER::printFields(std::ostream &os, int /*indent*/)
{

    return os;
}

void VrmlNodeCOVER::eventIn(double timeStamp,
                            const char *eventName,
                            const VrmlField *fieldValue)
{
    if ((strcmp(eventName, "set_soundEnvironment") == 0) || (strcmp(eventName, "soundEnvironment") == 0))
    {
        Player *player = System::the->getPlayer();
        if (player)
        {
            player->setEAXEnvironment(d_soundEnvironment.get());
        }
    }
    else if ((strcmp(eventName, "set_animationTimeStep") == 0) || (strcmp(eventName, "animationTimeStep") == 0))
    {
        System::the->setTimeStep(d_animationTimeStep.get());
    }
    else if ((strcmp(eventName, "set_activePerson") == 0) || (strcmp(eventName, "activePerson") == 0))
    {
        System::the->setActivePerson(d_activePerson.get());
    }
    else if ((strcmp(eventName, "set_saveTimestamp") == 0) || (strcmp(eventName, "saveTimestamp") == 0))
    {
        System::the->saveTimestamp(d_saveTimestamp.get());
    }
	else if ((strcmp(eventName, "set_loadPlugin") == 0) || (strcmp(eventName, "loadPlugin") == 0))
	{
		System::the->loadPlugin(d_loadPlugin.get());
	}
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}

// Set the value of one of the node fields.

void VrmlNodeCOVER::setField(const char *fieldName,
                             const VrmlField &fieldValue)
{
    if
        TRY_FIELD(soundEnvironment, SFInt)
    else if
        TRY_FIELD(animationTimeStep, SFInt)
    else if
        TRY_FIELD(activePerson, SFInt)
    else if
        TRY_FIELD(position1, SFVec3f)
    else if
        TRY_FIELD(position2, SFVec3f)
    else if
        TRY_FIELD(position3, SFVec3f)
    else if
        TRY_FIELD(position4, SFVec3f)
    else if
        TRY_FIELD(position5, SFVec3f)
    else if
        TRY_FIELD(position6, SFVec3f)
    else if
        TRY_FIELD(position7, SFVec3f)
    else if
        TRY_FIELD(position8, SFVec3f)
    else if
        TRY_FIELD(position9, SFVec3f)
    else if
        TRY_FIELD(position10, SFVec3f)
    else if
        TRY_FIELD(position11, SFVec3f)
    else if
        TRY_FIELD(position12, SFVec3f)
    else if
        TRY_FIELD(position13, SFVec3f)
    else if
        TRY_FIELD(position14, SFVec3f)
    else if
        TRY_FIELD(position15, SFVec3f)
    else if
        TRY_FIELD(orientation1, SFRotation)
    else if
        TRY_FIELD(orientation2, SFRotation)
    else if
        TRY_FIELD(orientation3, SFRotation)
    else if
        TRY_FIELD(orientation4, SFRotation)
    else if
        TRY_FIELD(orientation5, SFRotation)
    else if
        TRY_FIELD(orientation6, SFRotation)
    else if
        TRY_FIELD(orientation7, SFRotation)
    else if
        TRY_FIELD(orientation8, SFRotation)
    else if
        TRY_FIELD(orientation9, SFRotation)
    else if
        TRY_FIELD(orientation10, SFRotation)
    else if
        TRY_FIELD(orientation11, SFRotation)
    else if
        TRY_FIELD(orientation12, SFRotation)
    else if
        TRY_FIELD(orientation13, SFRotation)
    else if
        TRY_FIELD(orientation14, SFRotation)
    else if
        TRY_FIELD(orientation15, SFRotation)
	else if
		TRY_FIELD(loadPlugin, SFString)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
    if (strcmp("soundEnvironment", fieldName) == 0)
    {
        Player *player = System::the->getPlayer();
        if (player)
        {
            player->setEAXEnvironment(d_soundEnvironment.get());
        }
    }
    else if (strcmp("animationTimeStep", fieldName) == 0)
    {
        System::the->setTimeStep(d_animationTimeStep.get());
    }
    else if (strcmp("activePerson", fieldName) == 0)
    {
        System::the->setActivePerson(d_activePerson.get());
    }
	else if ((strcmp(fieldName, "set_loadPlugin") == 0) || (strcmp(fieldName, "loadPlugin") == 0))
	{
		System::the->loadPlugin(d_loadPlugin.get());
	}
}

// process remote key events, called by eventQueue
void VrmlNodeCOVER::remoteKeyEvent(enum VrmlNodeCOVER::KeyEventType type, const char *keystring)
{
    double timeStamp = System::the->time();
    if (type == KeyPress)
    {
        //cerr << "press " << keystring << endl;
        d_keyPressed.set(keystring);
        eventOut(timeStamp, "keyPressed", d_keyPressed);
    }
    else if (type == KeyRelease)
    {
        //cerr << "release " << keystring << endl;
        d_keyReleased.set(keystring);
        eventOut(timeStamp, "keyReleased", d_keyReleased);
    }
}

// process Key events
void VrmlNodeCOVER::keyEvent(enum VrmlNodeCOVER::KeyEventType type, const char *keyString)
{
    double timeStamp = System::the->time();

    if (type == Press)
    {
        d_keyPressed.set(keyString);
        eventOut(timeStamp, "keyPressed", d_keyPressed);
        d_localKeyPressed.set(keyString);
        eventOut(timeStamp, "localKeyPressed", d_keyPressed);
    }
    else if (type == Release)
    {
        d_keyReleased.set(keyString);
        eventOut(timeStamp, "keyReleased", d_keyReleased);
        d_localKeyReleased.set(keyString);
        eventOut(timeStamp, "localKeyReleased", d_keyReleased);
    }
    //fprintf(stderr, "theCOVER: key type=%d, string=%s\n", type, keystringMod);
    if (scene())
    {
        scene()->getSensorEventQueue()->sendKeyEvent(type, keyString);
    }
}
