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

#include <cassert>

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

void VrmlNodeCOVER::initFields(VrmlNodeCOVER *node, VrmlNodeType *t)
{
    for (size_t i = 0; i < NUM_POSITIONS; i++)
    {
        initFieldsHelper(node, t,
                         exposedField("position" + std::to_string(i + 1), node->d_positions[i]),
                         exposedField("orientation" + std::to_string(i + 1), node->d_orientations[i]));
    }
    initFieldsHelper(node, t,
                        exposedField("soundEnvironment", node->d_soundEnvironment, [](auto fieldValue) {
                            Player *player = System::the->getPlayer();
                            if (player)
                            {
                                player->setEAXEnvironment(theCOVER->d_soundEnvironment.get());
                            }
                        }),
                        exposedField("animationTimeStep", node->d_animationTimeStep, [](auto fieldValue) {
                            System::the->setTimeStep(theCOVER->d_animationTimeStep.get());
                        }),
                        exposedField("activePerson", node->d_activePerson, [](auto fieldValue){
                            System::the->setActivePerson(theCOVER->d_activePerson.get());
                        }),
                        exposedField("loadPlugin", node->d_loadPlugin, [](auto fieldValue){
                            System::the->loadPlugin(theCOVER->d_loadPlugin.get());
                        }),
                        exposedField("set_loadPlugin", node->d_loadPlugin, [](auto fieldValue){
                            System::the->loadPlugin(theCOVER->d_loadPlugin.get());
                            //this could be a hack to make the plugin load immediately or it is a mistake
                        }));
                        
    if(t)
    {
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
        t->addEventIn("saveTimestamp", VrmlField::SFSTRING);
    }
    VrmlNodeChild::initFields(node, t);
}

const char *VrmlNodeCOVER::name() { return "COVER"; }



VrmlNodeCOVER::VrmlNodeCOVER(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
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
    for (size_t i = 0; i < NUM_POSITIONS; i++)
    {
        d_positions[i].set(1000000, 1000000, 1000000);
        d_orientations[i].set(0, 1, 0, 0);
    }
    if(!theCOVER)
    {
        theCOVER = this;
    } else  
        assert(false);
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

void VrmlNodeCOVER::update(double timeNow)
{
    double tmpRot[16], tmpTrans[16];

    for (size_t i = 0; i < NUM_POSITIONS; i++)
    {
        Mrotation(tmpRot, d_orientations[i].get(), d_orientations[i].r());
        Mtrans(tmpTrans, d_positions[i].get());
        Mmult(transformations[i], tmpRot, tmpTrans);
    }
    
    
    //for(int u=0;u<4;u++)
    //cerr << "vrml:" << transformations[0][u][0] << " "transformations[0][u][1] << " "transformations[0][u][2] << " "transformations[0][u][3] << " " << endl;
    //cerr << transformations[0][u][0] << " "transformations[0][u][1] << " "transformations[0][u][2] << " "transformations[0][u][3] << " " << endl;
    //cerr << transformations[0][u][0] << " "transformations[0][u][1] << " "transformations[0][u][2] << " "transformations[0][u][3] << " " << endl;


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
