/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coPinEditor.h"
#include "coHSVPin.h"
#include "coAlphaHatPin.h"
#include "coAlphaBlankPin.h"
#include "coPreviewCube.h"
#include "VolumePlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRCollaboration.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <cover/coIntersection.h>
#include <virvo/vvtransfunc.h>
#include "coDefaultFunctionEditor.h"
#include "coHSVSelector.h"
#include <virvo/vvtoolshed.h>
#include <stdlib.h>

#include <OpenVRUI/coColoredBackground.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/osg/OSGVruiTransformNode.h>
#include <OpenVRUI/osg/OSGVruiHit.h>
#include <OpenVRUI/osg/OSGVruiPresets.h>
#include <OpenVRUI/util/vruiLog.h>

#include <osg/CullFace>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/TexEnv>

#include <osgDB/ReadFile>
#include <osgUtil/IntersectVisitor>

#include <config/CoviseConfig.h>

#include <algorithm>

#define ZOFFSET 0.1
#define TEXTURE_RES 64
#define TEXTURE_RES_COLOR 256

const float ColorPinSeparationY = 0.5f;

using namespace std;
using namespace osg;
using namespace vrui;
using namespace opencover;

coPinEditor::coPinEditor(vvTransFunc *transFunc, coDefaultFunctionEditor *functionEditor)
    : vruiCollabInterface(VolumeCoim.get(), "PinEditor", vruiCollabInterface::PinEditor)
{
    A = 0.5;
    B = 0.4;
    W = 80;
    H = 40;
    pickThreshold = 0.2;
    moveThreshold = 20;
    doMove = false;
    backgroundMode = 1;
    pickTime = -1;
    pickCoordX = pickCoordY = 0;
    mixChannelsActive = false;
    SELH = 3;
    COLORH = 5;
    OFFSET = 1.0;
    selectedRegion = -1;
    mode = SELECTION;
    myTransFunc = transFunc;
    myFunctionEditor = functionEditor;
    unregister = false;
    createLists();
    createSelectionBarLists();
    myDCS = new OSGVruiTransformNode(new MatrixTransform());
    myDCS->getNodePtr()->asGroup()->addChild(createBackgroundGroup().get());
    myDCS->getNodePtr()->asGroup()->addChild(createSelectionBarGeode().get());
    pinDCS = new MatrixTransform();
    myDCS->getNodePtr()->asGroup()->addChild(pinDCS.get());
    Matrix pinMatrix;
    pinMatrix.identity();
    pinMatrix.setTrans(A + B, -(A + B + COLORH + A + B + SELH + A + B), 0);
    pinDCS->setMatrix(pinMatrix);
    coIntersection::getIntersectorForAction("coAction")->add(myDCS, this);
    setPos(15, -4);
    updatePinList();
    currentScalarLabel = new coLabel();
    currentScalarLabel->setFontSize(5);
    currentScalarLabel->setString("x");
    labelBackground = new coColoredBackground(coUIElement::ITEM_BACKGROUND_NORMAL, coUIElement::ITEM_BACKGROUND_HIGHLIGHTED, coUIElement::ITEM_BACKGROUND_DISABLED);
    labelBackground->setPos(15, -59, 2);
    labelBackground->setZOffset(0.5);
    labelBackground->addElement(currentScalarLabel);
    myFunctionEditor->panel->addElement(labelBackground);
    labelBackground->setVisible(false);
    interactionA = new coCombinedButtonInteraction(coInteraction::ButtonA, "PinEditor", coInteraction::Menu);
    interactionB = new coCombinedButtonInteraction(coInteraction::ButtonB, "PinEditor", coInteraction::Menu);
    interactionC = new coCombinedButtonInteraction(coInteraction::ButtonC, "PinEditor", coInteraction::Menu);
}

void coPinEditor::init()
{
    mode = ADD_PIN;
    setMode(SELECTION);
}

coPinEditor::~coPinEditor()
{
    delete interactionA;
    delete interactionB;
    delete interactionC;

    for (list<coPin *>::iterator pin = pinList.begin(); pin != pinList.end(); ++pin)
        delete *pin;

    pinList.clear();

    coIntersection::getIntersectorForAction("coAction")->remove(this);
    myDCS->removeAllChildren();
    myDCS->removeAllParents();
    delete myDCS;
}

void coPinEditor::setTransFuncPtr(vvTransFunc *tf)
{
    for (list<coPin *>::iterator pin = pinList.begin(); pin != pinList.end(); ++pin)
        delete *pin;

    pinList.clear();

    selectedRegion = -1;
    setMode(SELECTION);
    currentPin = NULL;

    myTransFunc = tf;

    updatePinList();
}

void coPinEditor::createGeometry() {}
void coPinEditor::resizeGeometry() {}

void coPinEditor::remoteLock(const char *message)
{
    vruiCollabInterface::remoteLock(message);
}

void coPinEditor::remoteOngoing(const char *message)
{
    coPin *myPin = currentPin;
    if ((currentPin == NULL) || (currentPin->getID() != remoteContext))
    { // search for the remote Pin
        myPin = findPin(remoteContext);
    }

    if (myPin == NULL)
    {
        cerr << "pin not found ID: " << remoteContext << endl;
    }
    (void)message;
#if 0
   switch(message[0])
   {
      case 'M':
         if(myPin)
         {
            float position;
            sscanf(message,"M%f", &position);
            myPin->setPos(position);
            //myTransFunc->pins.move(myPin->jPin,position);
            updateColorBar();
            sortPins();
         }
         break;
      case 'X':
         if(myPin)
         {
            float position;
            sscanf(message,"X%f", &position);

            myPin->setPos(position);
            float h,s,v;
            myTransFunc->pins.getHSB(position,&h,&s,&v);
            //myTransFunc->pins.move(myPin->jPin,position);
            vvTFColor *cPin = dynamic_cast<vvTFColor *>(myPin->jPin);
            if(cPin)
            {
               if(myTransFunc->pins.getColorModel()==vvTFWidget::HSB_MODEL)
               {
                  cPin->setColors(h, s, v);
               }
               else if(myTransFunc->pins.getColorModel()==vvTFWidget::RGB_MODEL)
               {
                  float r,g,b;
                  vvToolshed::HSBtoRGB(h,s,v, &r,&g,&b);
                  cPin->setColors(r,g,b);
               }
               ((coHSVPin*)myPin)->setColor(h,s,v);
               myFunctionEditor->hsvSel->setColorHSB(h,s,v);
               myFunctionEditor->brightness->setValue(v);
            }
            else
            {
               updateColorBar();
            }
            sortPins();
         }
         break;
      case 'A':
         if(myPin)
         {
            float val;
            sscanf(message,"A%f", &val);
            setMax(val,remoteContext);
            if(currentPin->getID()==remoteContext)
               myFunctionEditor->max->setValue(val);
            updateColorBar();
         }
         break;
      case 'S':
         if(myPin)
         {
            float val;
            sscanf(message,"S%f", &val);
            setSlope(val,remoteContext);
            if(currentPin->getID()==remoteContext)
               myFunctionEditor->slope->setValue(val);
            updateColorBar();
         }
         break;
      case 'W':
         if(myPin)
         {
            float val;
            sscanf(message,"W%f", &val);
            setWidth(val,remoteContext);
            if(currentPin->getID()==remoteContext)
               myFunctionEditor->width->setValue(val);
            updateColorBar();
         }
         break;
      case 'N':
      {
         int id,type;
         sscanf(message,"N%d %d", &id,&type);
         EditMode saveMode = mode;
         myPin = currentPin;
         addPin(type,0);
         setMode(saveMode);
         currentPin=myPin;
         remoteContext=id;
      }
      break;
      case 'D':
      {
         int id;
         sscanf(message,"D%d", &id);
         deletePin(id);
      }
      break;
      case 'E':
      {
         int id;
         sscanf(message,"E%d", &id);
         myPin = currentPin;
         if((currentPin==NULL)||(currentPin->getID()!=id))
         {                                        // search for the remote Pin
            currentPin = findPin(id);
            highlightCurrent();
            adjustSelectionBar();
         }
         break;
         default:
         {
            cerr << "coPinEditor::Unknown remote Command " << endl;
         }
         break;
      }
   }
#endif
}

void coPinEditor::releaseRemoteLock(const char *message)
{
    vruiCollabInterface::releaseRemoteLock(message);
}

void coPinEditor::setPos(float x, float y, float)
{
    myX = x;
    myY = y;
    myDCS->setTranslation(x, y + getHeight(), 0.0);
}

vruiTransformNode *coPinEditor::getDCS()
{
    return myDCS;
}

void coPinEditor::setBrightness(float v, int context)
{
    coPin *myPin = currentPin;
    if (context != -1)
    {
        // remote interaction
        if ((currentPin == NULL) || (currentPin->getID() != context))
        { // search for the remote Pin
            myPin = findPin(context);
        }
    }

    if (myPin)
    {
        vvTFColor *colPin = dynamic_cast<vvTFColor *>(myPin->jPin);
        if (colPin)
        {
            float h, s, dummy;
            colPin->_col.getHSB(h, s, dummy);
            colPin->_col.setHSB(h, s, v);
            ((coHSVPin *)myPin)->setColor(h, s, v);
            updateColorBar();
        }
    }
}

void coPinEditor::setColor(float h, float s, float v, int context)
{
    coPin *myPin = currentPin;
    if (context != -1)
    {
        // remote interaction
        if ((currentPin == NULL) || (currentPin->getID() != context))
        { // search for the remote Pin
            myPin = findPin(context);
        }
    }

    if (myPin)
    {
        vvTFColor *colPin = dynamic_cast<vvTFColor *>(myPin->jPin);
        if (colPin)
        {
            colPin->_col.setHSB(h, s, v);
        }
        ((coHSVPin *)myPin)->setColor(h, s, v);
        updateColorBar();
    }
}

void coPinEditor::setTopWidth(float s, int context)
{
    coPin *myPin = currentPin;
    if (context != -1)
    {
        // remote interaction
        if ((currentPin == NULL) || (currentPin->getID() != context))
        { // search for the remote Pin
            myPin = findPin(context);
        }
    }

    if (myPin)
    {
        vvTFPyramid *pyr = dynamic_cast<vvTFPyramid *>(myPin->jPin);
        if (pyr)
        {
            ((coAlphaHatPin *)myPin)->setTopWidth(fabs(s),
                                                  myFunctionEditor->getMin(),
                                                  myFunctionEditor->getMax());
        }
        updateColorBar();
    }
}

void coPinEditor::setBotWidth(float w, int context)
{
    coPin *myPin = currentPin;
    if (context != -1)
    {
        // remote interaction
        if ((currentPin == NULL) || (currentPin->getID() != context))
        { // search for the remote Pin
            myPin = findPin(context);
        }
    }

    if (myPin)
    {
        if (coAlphaHatPin *pin = dynamic_cast<coAlphaHatPin *>(myPin))
        {
            pin->setBotWidth(w, myFunctionEditor->getMin(), myFunctionEditor->getMax());
        }
        else if (coAlphaBlankPin *pin = dynamic_cast<coAlphaBlankPin *>(myPin))
        {
            pin->setWidth(w, myFunctionEditor->getMin(), myFunctionEditor->getMax());
        }
        updateColorBar();
    }
}

void coPinEditor::setMax(float m, int context)
{
    coPin *myPin = currentPin;
    if (context != -1)
    {
        // remote interaction
        if ((currentPin == NULL) || (currentPin->getID() != context))
        { // search for the remote Pin
            myPin = findPin(context);
        }
    }
    if (myPin)
    {
        if (coAlphaHatPin *pin = dynamic_cast<coAlphaHatPin *>(myPin))
        {
            pin->setMax(m, myFunctionEditor->getMin(), myFunctionEditor->getMax());
        }
        updateColorBar();
    }
}

int coPinEditor::hit(vruiHit *hit)
{
    if (coVRCollaboration::instance()->getSyncMode() == coVRCollaboration::MasterSlaveCoupling
        && !coVRCollaboration::instance()->isMaster())
        return ACTION_DONE;

    if (!interactionA->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionA);
        interactionA->setHitByMouse(hit->isMouseHit());
    }
    if (!interactionB->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionB);
        interactionB->setHitByMouse(hit->isMouseHit());
    }
    if (!interactionC->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(interactionC);
        interactionC->setHitByMouse(hit->isMouseHit());
    }

    osgUtil::LineSegmentIntersector::Intersection osgHit = dynamic_cast<OSGVruiHit *>(hit)->getHit();

    static char message[100];

    float x = 0.f, y = 0.f;
    if (osgHit.drawable.valid())
    {
        Vec3 point = osgHit.getLocalIntersectPoint();
        x = (point[0] - (A + B)) / W;
        y = 1 + ((point[1] + (A + B)) / H);
        if (x > 1)
            x = 1;
        if (x < 0)
            x = 0;
        if (y > 1)
            y = 1;
        if (y < 0)
            y = 0;

        // adjust interaction elements on hat pins
        for (list<coPin *>::iterator pin = pinList.begin(); pin != pinList.end(); ++pin)
        {
            coAlphaHatPin *alphaHatPin = dynamic_cast<coAlphaHatPin *>(*pin);
            if (alphaHatPin)
            {
                float bottom = alphaHatPin->getBotWidth01();
                float top = alphaHatPin->getTopWidth01();
                float pos = alphaHatPin->getPos01();

                float d = (bottom - top) / 4. + top / 2.;
                float trans = -d;
                if (d < 0.1 && pos >= 0.0 && pos <= 1.0)
                {
                    trans = 0.;
                }
                else
                {
                    for (int i = 0; i < 2; i++)
                    {
                        if (pos + trans < 0.0 && pos + trans <= 1.0)
                        {
                            trans += d;
                            continue;
                        }
                        if (pos + trans > 1.0 && pos + trans - d >= 0.0)
                        {
                            trans -= d;
                            break;
                        }

                        if (fabs(pos + trans + d - x) < fabs(pos + trans - x)
                            && pos + trans + d <= 1.00001)
                        {
                            trans += d;
                        }
                    }
                }

                (*pin)->setHandleTrans(trans);
            }
            //cerr << "x: " << x << " current: " << pin->jPin->x << " dist: " <<fabs(pin->jPin->x - x) << endl;
        }

        if ((interactionA->getState() == coInteraction::Idle) && (interactionB->getState() == coInteraction::Idle) && (interactionC->getState() == coInteraction::Idle))
        {
            //VRUILOG(w1 << " " << w2 << " " << w3 << " " << w4)
            (*selectionBarColor)[0].set(0.2f, 0.2f, 0.2f, 1.0f);
            (*selectionBarColor)[1].set(0.5f, 0.5f, 0.5f, 1.0f);
            (*selectionBarColor)[2].set(0.2f, 0.2f, 0.2f, 1.0f);
            (*selectionBarColor)[3].set(0.5f, 0.5f, 0.5f, 1.0f);
            selectedRegion = -1;
            if (x > w3 + w2 + w1)
            {
                selectedRegion = 3;
                (*selectionBarColor)[3].set(1.0f, 0.0f, 0.0f, 1.0f);
            }
            else if (x > w2 + w1)
            {
                selectedRegion = 2;
                (*selectionBarColor)[2].set(1.0f, 0.0f, 0.0f, 1.0f);
            }
            else if (x > w1)
            {
                selectedRegion = 1;
                (*selectionBarColor)[1].set(1.0f, 0.0f, 0.0f, 1.0f);
            }
            else
            {
                selectedRegion = 0;
                (*selectionBarColor)[0].set(1.0f, 0.0f, 0.0f, 1.0f);
            }
            //VRUILOG("coPinEditor::hit info: selectedRegion = " << selectedRegion)
			selectionBarColor->dirty();
            selectionBarGeometry->dirtyDisplayList();
        }
    }

    if (mode == ADD_PIN)
    {
        float valuex = x - currentPin->handleTrans();
        x = ts_clamp(x, 0.0f, 1.0f);
        valuex = virvo::lerp(myFunctionEditor->getMin(), myFunctionEditor->getMax(), x);
        currentPin->setPos(valuex,
                           myFunctionEditor->getMin(),
                           myFunctionEditor->getMax());
        sprintf(message, "X%f %f %f", x, .5, .5);
        sendOngoingMessage(message);
        vvTFColor *col = dynamic_cast<vvTFColor *>(currentPin->jPin);
        if (col)
        {
            vvColor vc = myTransFunc->computeColor(x, .5, .5);
            float h, s, v;
            col->_col = vc;
            vc.getHSB(h, s, v);
            ((coHSVPin *)currentPin)->setColor(h, s, v);
            myFunctionEditor->hsvSel->setColorHSB(h, s, v);
            myFunctionEditor->brightness->setValue(v);
        }
        else
        {
            updateColorBar();
        }
        sortPins();
    }

    // pin selektieren nach Ablauf der Zeit oder nach grosser Bewegung

    if (!mixChannelsActive && interactionA->isRunning() && (!doMove))
    {
        if ((pickTime > 0) && ((cover->frameTime() - pickTime) > pickThreshold))
        {
            doMove = true;
            pickTime = -1;
            selectPin(pickCoordX, pickCoordY);
            //cerr << "Lock ID="<< currentPin->getID() << endl;
            sprintf(message, "%d", currentPin->getID());
            sendLockMessage(message);
        }
        if (fabs(x - pickCoordX) > moveThreshold
            || (currentPin
                && ((y < ColorPinSeparationY && dynamic_cast<vvTFColor *>(currentPin->jPin))
                    || (y >= ColorPinSeparationY && (dynamic_cast<vvTFPyramid *>(currentPin->jPin) || dynamic_cast<vvTFSkip *>(currentPin->jPin))))))
        {
            doMove = true;
            pickTime = -1;
            selectPin(pickCoordX, pickCoordY);
            //cerr << "Lock ID="<< currentPin->getID() << endl;
            sprintf(message, "%d", currentPin->getID());
            sendLockMessage(message);
        }
    }

    if (interactionA->wasStarted() || interactionB->wasStarted() || interactionC->wasStarted())
    {
        if (mode != ADD_PIN)
            myFunctionEditor->putUndoBuffer();
        if (currentPin != NULL)
        {
            //cerr << "Lock ID="<< currentPin->getID() << endl;
            sprintf(message, "%d", currentPin->getID());
            sendLockMessage(message);
        }
    }
    if (interactionA->wasStarted())
    {
        if (mode == ADD_PIN)
        {
            if (currentPin == NULL)
                return ACTION_CALL_ON_MISS;
            currentPin->select();
            vvTFColor *colPin = dynamic_cast<vvTFColor *>(currentPin->jPin);
            vvTFPyramid *pyrPin = dynamic_cast<vvTFPyramid *>(currentPin->jPin);
            vvTFSkip *skipPin = dynamic_cast<vvTFSkip *>(currentPin->jPin);
            if (colPin)
            {
                setMode(EDIT_COLOR);
            }
            else if (pyrPin)
            {
                myFunctionEditor->topWidth->setValue(pyrPin->_top[0]);
                myFunctionEditor->botWidth->setValue(pyrPin->_bottom[0]);
                myFunctionEditor->max->setValue(pyrPin->_opacity);
                setMode(EDIT_HAT);
            }
            else if (skipPin)
            {
                myFunctionEditor->botWidth->setValue(skipPin->_size[0]);
                setMode(EDIT_BLANK);
            }
            myFunctionEditor->endPinCreation();
        }
        else
        {
            // don't select here but later, when the mousebutton was released
            // or when the pointer moved a lot
            // what we do is remember the x position and the current time
            if (isNearestSelected(x, y))
            {
                doMove = true;
            }
            else
            {
                pickTime = cover->frameTime();
                pickCoordX = x;
                pickCoordY = y;
                doMove = false;
            }
        }
    }

    if (interactionA->isRunning())
    {
        // left Button is moved
        if (doMove)
        {
            if (currentPin != NULL)
            {
                sprintf(message, "M%f", x);
                sendOngoingMessage(message);
                float valuex = x - currentPin->handleTrans();
                x = ts_clamp(x, 0.0f, 1.0f);
                valuex = virvo::lerp(myFunctionEditor->getMin(), myFunctionEditor->getMax(), x);
                currentPin->setPos(valuex,
                                   myFunctionEditor->getMin(),
                                   myFunctionEditor->getMax());
                updateColorBar();
                sortPins();
            }
        }
    }

    if (!mixChannelsActive && (interactionB->wasStarted() || interactionC->wasStarted()))
    {
        coCoord mouseCoord = cover->getPointerMat();
        lastRoll = mouseCoord.hpr[2];

        if (currentPin == NULL)
        {
            selectPin(x, y);
        }
        if (currentPin == NULL)
            return ACTION_CALL_ON_MISS;
    }

    return ACTION_CALL_ON_MISS;
}

void coPinEditor::miss()
{
    //abbort creation if currently in Creation Mode
    /*if(mode == ADD_PIN)
     {
         setMode(SELECTION);
         myFunctionEditor->endPinCreation();
         deleteCurrentPin();
     }*/
    unregister = true;
}

void coPinEditor::update()
{
    char formats[2][16] = {
        { "%.2f" },
        { "%.0f" }
    };
    char *formatString;
    float scalarRange;

    //cerr << "pickTime:" << pickTime << " doMove:"<< doMove << " getButtonStatus:" << cover->getButton()->getButtonStatus() << endl;
    if (!interactionA->isRunning())
    {
        if (!doMove && pickTime > 0)
        { // button was released before pin was selected, so do it now
            doMove = false;
            pickTime = -1;
            selectPin(pickCoordX, pickCoordY);
        }
        if (doMove || pickTime > 0)
        {
            doMove = false;
            pickTime = -1;
        }
    }

    if (interactionA->wasStopped() || interactionB->wasStopped() || interactionC->wasStopped())
    {
        sendReleaseMessage(NULL);
    }

    // update CurrentPin label

    if (currentPin)
    {
        static float oldX = -11111;
        
        float pos = currentPin->getPos01();

        labelBackground->setVisible(true);
        if (oldX != pos + currentPin->handleTrans())
        {
            oldX = pos;
            char num[50];
            scalarRange = myFunctionEditor->getMax() - myFunctionEditor->getMin() + 1.0f;
            if (scalarRange == 256.0f || scalarRange == 4096.0f)
                formatString = formats[1];
            else
                formatString = formats[0];
            sprintf(num, formatString, myFunctionEditor->getMin() + (oldX + currentPin->handleTrans()) * (myFunctionEditor->getMax() - myFunctionEditor->getMin()));
            currentScalarLabel->setString(num);
            labelBackground->setPos(15 + ((pos + currentPin->handleTrans()) * W) - currentScalarLabel->getWidth() / 2.0, -59, 2);
        }
    }
    else
    {
        labelBackground->setVisible(false);
    }

    static int oldSyncMode = -1;
    if (oldSyncMode != coVRCollaboration::instance()->getSyncMode())
    {
        if (currentPin)
        {
            if (oldSyncMode == coVRCollaboration::LooseCoupling
                && coVRCollaboration::instance()->isMaster())
            {
                static char message[100];
                sprintf(message, "E%d", currentPin->getID());
                sendOngoingMessage(message);
            }
        }
        oldSyncMode = coVRCollaboration::instance()->getSyncMode();
    }
    if (coVRCollaboration::instance()->getSyncMode() == coVRCollaboration::MasterSlaveCoupling
        && !coVRCollaboration::instance()->isMaster())
        return;
    if (interactionB->isRunning())
    {
        // middle Button is pressed
        coCoord mouseCoord = cover->getPointerMat();
        float myValue = 0;
        if (currentPin)
        {
            vvTFPyramid *pyrPin = dynamic_cast<vvTFPyramid *>(currentPin->jPin);
            vvTFSkip *skipPin = dynamic_cast<vvTFSkip *>(currentPin->jPin);
            if (pyrPin)
            {
                if (selectedRegion == 1)
                {
                    myValue = pyrPin->_top[0];
                }
                else
                {
                    myValue = pyrPin->_bottom[0];
                }
            }
            else if (skipPin)
            {
                myValue = skipPin->_size[0];
            }
            if (lastRoll != mouseCoord.hpr[2])
            {
                float lastValue = myValue;
                if ((lastRoll - mouseCoord.hpr[2]) > 180)
                    lastRoll -= 360;
                if ((lastRoll - mouseCoord.hpr[2]) < -180)
                    lastRoll += 360;
                cerr << "MyValue1: " << myValue << " lastRoll: " << lastRoll << " mouseCoord.hpr[2]: " << mouseCoord.hpr[2] << endl;

                if ((selectedRegion != 1) && pyrPin)
                {
                    myValue = tan(atan(myValue) + (lastRoll - mouseCoord.hpr[2]) / 90.0);
                    //fprintf(stderr, "slope=%f\n", myValue);
                } // slope
                else
                {
                    myValue -= (lastRoll - mouseCoord.hpr[2]) / 90.0;
                }
                //cerr << "MyValue2: " << myValue << endl;
                lastRoll = mouseCoord.hpr[2];
                static char message[100];
                if (lastValue != myValue)
                {
                    if (pyrPin)
                    {
                        if (selectedRegion != 1)
                        {
                            if (myValue < 0.0)
                                myValue = 0.0;
                            ((coAlphaHatPin *)currentPin)->setTopWidth(myValue,
                                                                       myFunctionEditor->getMin(),
                                                                       myFunctionEditor->getMax());
                            myFunctionEditor->topWidth->setValue(myValue);
                            sprintf(message, "S%f", myValue);
                            sendOngoingMessage(message);
                        }
                        else
                        {
                            myValue = ts_clamp(myValue, myFunctionEditor->getMin(), myFunctionEditor->getMax());
                            ((coAlphaHatPin *)currentPin)->setBotWidth(myValue,
                                                                       myFunctionEditor->getMin(),
                                                                       myFunctionEditor->getMax());
                            myFunctionEditor->botWidth->setValue(myValue);
                            sprintf(message, "A%f", myValue);
                            sendOngoingMessage(message);
                        }
                    }
                    updateColorBar();
                }
            }
        }
    }

    if (unregister)
    {
        if (interactionA->isRegistered() && (interactionA->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionA);
        }
        if (interactionB->isRegistered() && (interactionB->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionB);
        }
        if (interactionC->isRegistered() && (interactionC->getState() != coInteraction::Active))
        {
            coInteractionManager::the()->unregisterInteraction(interactionC);
        }
        if ((!interactionA->isRegistered()) && (!interactionB->isRegistered()) && (!interactionC->isRegistered()))
        {
            unregister = false;
            (*selectionBarColor)[0].set(0.2f, 0.2f, 0.2f, 1.0f);
            (*selectionBarColor)[1].set(0.5f, 0.5f, 0.5f, 1.0f);
            (*selectionBarColor)[2].set(0.2f, 0.2f, 0.2f, 1.0f);
            (*selectionBarColor)[3].set(0.5f, 0.5f, 0.5f, 1.0f);
			selectionBarColor->dirty();
            selectionBarGeometry->dirtyDisplayList();
        }
    }
}

class coPinCompare
{
public:
    bool operator()(const coPin *, const coPin *) const;
};

bool coPinCompare::operator()(const coPin *p1, const coPin *p2) const
{
    return p1->getPosValue() < p2->getPosValue();
}

void coPinEditor::sortPins()
{
    // sort pinList;
    pinList.sort(coPinCompare());
}

bool coPinEditor::isNearestSelected(float x, float y)
{
    coPin *minPin = NULL;
    float minDist = 2;
    for (list<coPin *>::iterator pin = pinList.begin(); pin != pinList.end(); ++pin)
    {
        vvTFWidget *p = (*pin)->jPin;
        if (y > ColorPinSeparationY)
        {
            if (dynamic_cast<vvTFSkip *>(p) || dynamic_cast<vvTFPyramid *>(p))
                continue;
        }
        else
        {
            if (dynamic_cast<vvTFColor *>(p))
                continue;
        }

        float pos = (*pin)->getPos01();

        if (fabs(pos - x) < minDist)
        {
            minPin = (*pin);
            minDist = fabs(pos - x);
        }
    }

    if (minPin == currentPin)
    {
        return true;
    }
    return false;
}

void coPinEditor::selectPin(float x, float y)
{
    coPin *minPin = NULL;
    float minDist = 2;
    for (list<coPin *>::iterator pin = pinList.begin(); pin != pinList.end(); ++pin)
    {
        vvTFWidget *p = (*pin)->jPin;
        if (y > ColorPinSeparationY)
        {
            if (dynamic_cast<vvTFPyramid *>(p) || dynamic_cast<vvTFSkip *>(p))
                continue;
        }
        else
        {
            if (dynamic_cast<vvTFColor *>(p))
                continue;
        }

        float pos = (*pin)->getPos01();

        if (fabs(pos + (*pin)->handleTrans() - x) < minDist)
        {
            minPin = (*pin);
            minDist = fabs(pos + (*pin)->handleTrans() - x);
        }

        //cerr << "x: " << x << " current: " << pin->jPin->x << " dist: " <<fabs(pin->jPin->x - x) << endl;
    }

    if (minPin)
    {
        /*if((currentPin == minPin) && (mode != SELECTION))
        {
        pinList.find(currentPin);             // get the next nearest pin (either to the let, or to the right)
        cerr << "Current == newarest" << mode << endl;
        float dist1=2;
        pinList.next();
        if(pinList.current())
        {
        minPin = pinList.current();
        dist1= fabs(minPin->jPin->x - x);
        }
      pinList.prev();
      pinList.prev();
      if(pinList.current())
      {
      if(fabs(pinList.current()->jPin->x - x) < dist1)
      {
      minPin = pinList.current();
      }
      }
      }*/
        currentPin = minPin;
        highlightCurrent();
        if (coVRCollaboration::instance()->getSyncMode() != coVRCollaboration::LooseCoupling)
        {
            static char message[100];
            sprintf(message, "E%d", currentPin->getID());
            sendOngoingMessage(message);
        }
    }
    adjustSelectionBar();
}

void coPinEditor::highlightCurrent()
{
    for (list<coPin *>::iterator pin = pinList.begin(); pin != pinList.end(); ++pin)
    {
        (*pin)->deSelect();
    }

    currentPin->select();
    vvTFColor *colPin = dynamic_cast<vvTFColor *>(currentPin->jPin);
    vvTFPyramid *pyrPin = dynamic_cast<vvTFPyramid *>(currentPin->jPin);
    vvTFSkip *skipPin = dynamic_cast<vvTFSkip *>(currentPin->jPin);
    if (colPin)
    {
        float h, s, v;
        colPin->_col.getHSB(h, s, v);
        myFunctionEditor->hsvSel->setColorHSB(h, s, v);
        myFunctionEditor->brightness->setValue(v);
        setMode(EDIT_COLOR);
    }
    else if (pyrPin)
    {
        myFunctionEditor->topWidth->setValue(pyrPin->_top[0]);
        myFunctionEditor->botWidth->setValue(pyrPin->_bottom[0]);
        myFunctionEditor->max->setValue(pyrPin->_opacity);
        setMode(EDIT_HAT);
    }
    else if (skipPin)
    {
        myFunctionEditor->topWidth->setValue(skipPin->_size[0]);
        setMode(EDIT_BLANK);
    }
}

void coPinEditor::setMode(EditMode newMode)
{
    // Parameters for rotary knob layout:
    const float CENTER_X = 103.0;
    const float CENTER_Y = -43.0;
    const float DIST_Y_SMALL = 12.0;
    const float YOFFSET = 6;

    //cerr << "setMode:" << newMode<< endl;
    if (mode != newMode)
    {
        myFunctionEditor->panel->hide(myFunctionEditor->hsvSel);
        myFunctionEditor->panel->hide(myFunctionEditor->cube);
        myFunctionEditor->panel->hide(myFunctionEditor->topWidth);
        myFunctionEditor->panel->hide(myFunctionEditor->botWidth);
        myFunctionEditor->panel->hide(myFunctionEditor->max);
        myFunctionEditor->panel->hide(myFunctionEditor->brightness);
        myFunctionEditor->panel->hide(myFunctionEditor->mixChannelsButton);
        myFunctionEditor->panel->hide(myFunctionEditor->mixChannels01);

        if (newMode == ADD_PIN)
        {
            for (list<coPin *>::iterator pin = pinList.begin(); pin != pinList.end(); ++pin)
            {
                (*pin)->deSelect();
            }
        }
        else if (newMode == EDIT_COLOR)
        {
            myFunctionEditor->panel->show(myFunctionEditor->hsvSel);
            myFunctionEditor->panel->show(myFunctionEditor->cube);
            myFunctionEditor->panel->show(myFunctionEditor->brightness);
        }
        else if (newMode == EDIT_HAT)
        {
            myFunctionEditor->topWidth->setPos(CENTER_X, CENTER_Y + YOFFSET + DIST_Y_SMALL);
            myFunctionEditor->botWidth->setPos(CENTER_X, CENTER_Y + YOFFSET - DIST_Y_SMALL);
            myFunctionEditor->max->setPos(CENTER_X, CENTER_Y + YOFFSET - 3 * DIST_Y_SMALL);
            myFunctionEditor->panel->show(myFunctionEditor->topWidth);
            myFunctionEditor->panel->show(myFunctionEditor->botWidth);
            myFunctionEditor->panel->show(myFunctionEditor->max);
        }
        else if (newMode == EDIT_BLANK)
        {
            myFunctionEditor->botWidth->setPos(CENTER_X, CENTER_Y);
            myFunctionEditor->panel->show(myFunctionEditor->botWidth);
        }
        else
        {
            if (myFunctionEditor->getNumChannels() == 2)
            {
                myFunctionEditor->panel->show(myFunctionEditor->mixChannelsButton);
                myFunctionEditor->panel->show(myFunctionEditor->mixChannels01);
            }
        }
        mode = newMode;
    }
}

void coPinEditor::undoAddPin()
{
    setMode(SELECTION);
    myFunctionEditor->endPinCreation();
    deleteCurrentPin();
}

void coPinEditor::addPin(int type, int local)
{
    if (mode == ADD_PIN)
    {
        setMode(SELECTION);
        deleteCurrentPin();
    }
    setMode(ADD_PIN);
    vvTFWidget *jPin = NULL;
    switch (type)
    {
    case COLOR:
    {
        vvTFColor *colPin = new vvTFColor();
        jPin = colPin;
        vvColor col = myTransFunc->computeColor(jPin->pos()[0]);
        float h, s, v;
        col.getHSB(h, s, v);
        coHSVPin *pin = new coHSVPin(pinDCS.get(), H, W, colPin);
        colPin->_col = col;
        pin->setColor(h, s, v);
        currentPin = pin;
    }
    break;
    case ALPHA_HAT:
    {
        vvTFPyramid *pyrPin = new vvTFPyramid();
        pyrPin->setOwnColor(false);
        jPin = pyrPin;
        pyrPin->_top[0] = 0.f;
        pyrPin->_opacity = 1.f;
        pyrPin->_bottom[0] = 1.f;
        coAlphaHatPin *pin = new coAlphaHatPin(pinDCS.get(), H, W, pyrPin);
        currentPin = pin;
    }
    break;
    case ALPHA_BLANK:
    {
        vvTFSkip *skipPin = new vvTFSkip();
        jPin = skipPin;
        skipPin->_size[0] = 1. / 255.;
        currentPin = new coAlphaBlankPin(pinDCS.get(), H, W, skipPin);
    }
    }
    float x = myFunctionEditor->getMin();
    currentPin->setPos(x, myFunctionEditor->getMin(), myFunctionEditor->getMax());
    pinList.push_back(currentPin);
    if (jPin)
        myTransFunc->_widgets.push_back(jPin);
    if (local)
    {
        static char message[100];
        sprintf(message, "N%d %d", currentPin->getID(), type);
        sendOngoingMessage(message);
    }
}

void coPinEditor::deleteCurrentPin()
{
    if (mode == ADD_PIN)
    {
        setMode(SELECTION);
        myFunctionEditor->endPinCreation();
    }
    if (currentPin)
    {

        static char message[100];
        sprintf(message, "D%d", currentPin->getID());
        sendOngoingMessage(message);

        list<coPin *>::iterator pin = find(pinList.begin(), pinList.end(), currentPin);
        if (pin != pinList.end())
        {
            std::vector<vvTFWidget *>::iterator it = std::find(myTransFunc->_widgets.begin(),
                                                               myTransFunc->_widgets.end(), currentPin->jPin);
            delete *it;
            myTransFunc->_widgets.erase(it);
            pinList.erase(pin);
            delete currentPin;
            currentPin = NULL;
            updateColorBar();
        }
        else
        {
            fprintf(stderr, "deleteCurrentPin: notFound\n");
        }
        setMode(SELECTION);
        currentPin = NULL;
    }
}

void coPinEditor::deleteAllPins()
{
    list<coPin *>::iterator pin = pinList.begin();
    while (pin != pinList.end())
    {
        currentPin = *pin;
        std::vector<vvTFWidget *>::iterator it = std::find(myTransFunc->_widgets.begin(),
                                                           myTransFunc->_widgets.end(), currentPin->jPin);
        delete *it;
        myTransFunc->_widgets.erase(it);
        pin = pinList.erase(pin);
        delete currentPin;
    }
    currentPin = NULL;
    updateColorBar();
    setMode(SELECTION);
}

void coPinEditor::deletePin(int ID)
{
    if (mode == ADD_PIN)
    {
        if ((currentPin) && (currentPin->getID() == ID))
        {
            setMode(SELECTION);
            myFunctionEditor->endPinCreation();
        }
    }
    coPin *myPin = currentPin;
    if ((currentPin == NULL) || (currentPin->getID() != ID))
    { // search for the remote Pin
        myPin = findPin(ID);
    }

    if (myPin)
    {
        if (currentPin == myPin)
        {
            currentPin = NULL;
            setMode(SELECTION);
        }
        list<coPin *>::iterator pin = find(pinList.begin(), pinList.end(), myPin);
        if (pin != pinList.end())
        {
            std::vector<vvTFWidget *>::iterator it = std::find(myTransFunc->_widgets.begin(),
                                                               myTransFunc->_widgets.end(), myPin->jPin);
            delete *it;
            myTransFunc->_widgets.erase(it);
            delete myPin;
            pinList.erase(pin);

            updateColorBar();
        }
        else
        {
            fprintf(stderr, "deletePin: %d notFound\n", ID);
        }
    }
}

void coPinEditor::updateColorBar()
{
    myTransFunc->makeColorBar(TEXTURE_RES_COLOR, textureData, myFunctionEditor->getMin(), myFunctionEditor->getMax(), true);

    Image *image = tex->getImage();
    image->setImage(TEXTURE_RES_COLOR, 2, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                    textureData, Image::NO_DELETE, 4);
    image->dirty();
    myFunctionEditor->updateVolume();
    adjustSelectionBar();
}

void coPinEditor::updateBackground(unsigned char *backgroundTextureData)
{
    Image *image = histoTex->getImage();
    if (image == 0)
    {
        image = new Image();
        histoTex->setImage(image);
    }
    image->setImage(TEXTURE_RES_BACKGROUND, TEXTURE_RES_BACKGROUND, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, backgroundTextureData, Image::NO_DELETE, 3);
    image->dirty();
    backgroundGeode->setName("histogram");
    HistoBackgroundGeostate->setTextureAttributeAndModes(0, histoTex.get(), StateAttribute::ON);
}

void coPinEditor::setBackgroundType(int mo)
{
    if (mo != backgroundMode)
    {
        backgroundMode = mo;
        if (backgroundMode == 0)
        {
            backgroundGeode->setStateSet(HistoBackgroundGeostate.get());
            backgroundGeometry->setTexCoordArray(0, texcoordColor.get());
            texcoordColor->dirty();
        }
        else
        {
            backgroundGeode->setStateSet(NormalBackgroundGeostate.get());
            backgroundGeometry->setTexCoordArray(0, texcoord.get());
            texcoord->dirty();
        }
    }
    backgroundGeometry->dirtyDisplayList();
}

void coPinEditor::setMixChannelsActive(bool active)
{
    mixChannelsActive = active;
}

void coPinEditor::updatePinList()
{

    for (list<coPin *>::iterator pin = pinList.begin(); pin != pinList.end(); ++pin)
        delete *pin;

    pinList.clear();

    selectedRegion = -1;
    setMode(SELECTION);
    currentPin = NULL;
    float minv = myFunctionEditor->getMin();
    float maxv = myFunctionEditor->getMax();
    for (std::vector<vvTFWidget *>::const_iterator it = myTransFunc->_widgets.begin();
         it != myTransFunc->_widgets.end(); ++it)
    {
        if (vvTFColor *colPin = dynamic_cast<vvTFColor *>(*it))
        {
            float h, s, v;
            colPin->_col.getHSB(h, s, v);

            coHSVPin *pin = new coHSVPin(pinDCS.get(), H, W, colPin);
            pin->setColor(h, s, v);
            pin->setPos(colPin->pos()[0], minv, maxv);
            pinList.push_back(pin);
        }
        else if (vvTFPyramid *pyrPin = dynamic_cast<vvTFPyramid *>(*it))
        {
            coAlphaHatPin *pin = new coAlphaHatPin(pinDCS.get(), H, W, pyrPin);
            pin->setTopWidth(pyrPin->top()[0], minv, maxv);
            pin->setBotWidth(pyrPin->bottom()[0], minv, maxv);
            pin->setPos(pyrPin->pos()[0], minv, maxv);
            pinList.push_back(pin);
        }
        else if (vvTFSkip *skipPin = dynamic_cast<vvTFSkip *>(*it))
        {
            coAlphaBlankPin *pin = new coAlphaBlankPin(pinDCS.get(), H, W, skipPin);
            pin->setWidth(skipPin->size()[0], minv, maxv);
            pin->setPos(skipPin->pos()[0], minv, maxv);
            pinList.push_back(pin);
        }
    }
    updateColorBar();
}

void coPinEditor::createLists()
{

    color = new Vec4Array(1);
    coord = new Vec3Array(12);
    coordSel = new Vec3Array(12);
    coordColor = new Vec3Array(12);
    coordt = new Vec3Array(4);
    coordColort = new Vec3Array(4);
    normal = new Vec3Array(32);
    normalSel = new Vec3Array(24);
    normalt = new Vec3Array(1);
    texcoord = new Vec2Array(4);
    texcoordColor = new Vec2Array(4);

    ushort *verticesArray = new ushort[8 * 4];
    ushort *verticesArraySel = new ushort[6 * 4];

    (*coord)[0].set(0.0, -(SELH + A + B + COLORH + A + B), 0.0);
    (*coord)[1].set(2 * (A + B) + W, -(SELH + A + B + COLORH + A + B), 0.0);
    (*coord)[2].set(2 * (A + B) + W, -(2 * (A + B) + H + SELH + A + B + COLORH + A + B), 0.0);
    (*coord)[3].set(0.0, -(2 * (A + B) + H + SELH + A + B + COLORH + A + B), 0.0);
    (*coord)[4].set(A, -(A + SELH + A + B + COLORH + A + B), A);
    (*coord)[5].set(A + B + W + B, -(A + SELH + A + B + COLORH + A + B), A);
    (*coord)[6].set(A + B + W + B, -(A + B + H + B + SELH + A + B + COLORH + A + B), A);
    (*coord)[7].set(A, -(A + B + H + B + SELH + A + B + COLORH + A + B), A);
    (*coord)[8].set(A + B, -(A + B + SELH + A + B + COLORH + A + B), A - B);
    (*coord)[9].set(A + B + W, -(A + B + SELH + A + B + COLORH + A + B), A - B);
    (*coord)[10].set(A + B + W, -(A + B + H + SELH + A + B + COLORH + A + B), A - B);
    (*coord)[11].set(A + B, -(A + B + H + SELH + A + B + COLORH + A + B), A - B);

    (*coordColor)[0].set(0.0, 0.0, 0.0);
    (*coordColor)[1].set(2 * (A + B) + W, 0.0, 0.0);
    (*coordColor)[2].set(2 * (A + B) + W, -(2 * (A + B) + COLORH), 0.0);
    (*coordColor)[3].set(0.0, -(2 * (A + B) + COLORH), 0.0);
    (*coordColor)[4].set(A, -A, A);
    (*coordColor)[5].set(A + B + W + B, -A, A);
    (*coordColor)[6].set(A + B + W + B, -(A + B + COLORH + B), A);
    (*coordColor)[7].set(A, -(A + B + COLORH + B), A);
    (*coordColor)[8].set(A + B, -(A + B), A - B);
    (*coordColor)[9].set(A + B + W, -(A + B), A - B);
    (*coordColor)[10].set(A + B + W, -(A + B + COLORH), A - B);
    (*coordColor)[11].set(A + B, -(A + B + COLORH), A - B);

    (*coordSel)[0].set(0.0, -(COLORH + A + B), 0.0);
    (*coordSel)[1].set(2 * (A + B) + W, -(COLORH + A + B), 0.0);
    (*coordSel)[2].set(2 * (A + B) + W, -(2 * (A + B) + SELH + COLORH + A + B), 0.0);
    (*coordSel)[3].set(0.0, -(2 * (A + B) + SELH + COLORH + A + B), 0.0);
    (*coordSel)[4].set(A, -(A + COLORH + A + B), A);
    (*coordSel)[5].set(A + B + W + B, -(A + COLORH + A + B), A);
    (*coordSel)[6].set(A + B + W + B, -(A + B + SELH + B + COLORH + A + B), A);
    (*coordSel)[7].set(A, -(A + B + SELH + B + COLORH + A + B), A);
    (*coordSel)[8].set(A + B, -(A + B + COLORH + A + B), A - B);
    (*coordSel)[9].set(A + B + W, -(A + B + COLORH + A + B), A - B);
    (*coordSel)[10].set(A + B + W, -(A + B + SELH + COLORH + A + B), A - B);
    (*coordSel)[11].set(A + B, -(A + B + SELH + COLORH + A + B), A - B);

    (*texcoord)[0].set(0.0, 0.0);
    (*texcoord)[1].set(10.0, 0.0);
    (*texcoord)[2].set(10.0, 5.0);
    (*texcoord)[3].set(0.0, 5.0);

    (*texcoordColor)[0].set(0.0, 0.0);
    (*texcoordColor)[1].set(1.0, 0.0);
    (*texcoordColor)[2].set(1.0, 1.0);
    (*texcoordColor)[3].set(0.0, 1.0);

    (*coordt)[3].set(A + B, -(A + B + SELH + A + B + COLORH + A + B), A - B);
    (*coordt)[2].set(A + B + W, -(A + B + SELH + A + B + COLORH + A + B), A - B);
    (*coordt)[1].set(A + B + W, -(A + B + H + SELH + A + B + COLORH + A + B), A - B);
    (*coordt)[0].set(A + B, -(A + B + H + SELH + A + B + COLORH + A + B), A - B);

    (*coordColort)[3].set(A + B, -(A + B), A - B);
    (*coordColort)[2].set(A + B + W, -(A + B), A - B);
    (*coordColort)[1].set(A + B + W, -(A + B + COLORH), A - B);
    (*coordColort)[0].set(A + B, -(A + B + COLORH), A - B);

    (*color)[0].set(0.8f, 0.8f, 0.8f, 1.0f);

    float isqrtwo = 1.0 / sqrt(2.0);

    for (int i = 0; i < 4; ++i)
    {
        (*normal)[0 * 4 + i].set(0.0, isqrtwo, isqrtwo);
        (*normal)[1 * 4 + i].set(isqrtwo, 0.0, isqrtwo);
        (*normal)[2 * 4 + i].set(0.0, -isqrtwo, isqrtwo);
        (*normal)[3 * 4 + i].set(-isqrtwo, 0.0, isqrtwo);
        (*normal)[4 * 4 + i].set(0.0, -isqrtwo, isqrtwo);
        (*normal)[5 * 4 + i].set(-isqrtwo, 0.0, isqrtwo);
        (*normal)[6 * 4 + i].set(0.0, isqrtwo, isqrtwo);
        (*normal)[7 * 4 + i].set(isqrtwo, 0.0, isqrtwo);
    }

    (*normalt)[0].set(0.0, 0.0, 1.0);

    verticesArray[0] = 0;
    verticesArray[1] = 4;
    verticesArray[2] = 5;
    verticesArray[3] = 1;
    verticesArray[4] = 1;
    verticesArray[5] = 5;
    verticesArray[6] = 6;
    verticesArray[7] = 2;
    verticesArray[8] = 2;
    verticesArray[9] = 6;
    verticesArray[10] = 7;
    verticesArray[11] = 3;
    verticesArray[12] = 3;
    verticesArray[13] = 7;
    verticesArray[14] = 4;
    verticesArray[15] = 0;
    verticesArray[16] = 4;
    verticesArray[17] = 8;
    verticesArray[18] = 9;
    verticesArray[19] = 5;
    verticesArray[20] = 5;
    verticesArray[21] = 9;
    verticesArray[22] = 10;
    verticesArray[23] = 6;
    verticesArray[24] = 6;
    verticesArray[25] = 10;
    verticesArray[26] = 11;
    verticesArray[27] = 7;
    verticesArray[28] = 7;
    verticesArray[29] = 11;
    verticesArray[30] = 8;
    verticesArray[31] = 4;

    for (int i = 0; i < 4; ++i)
    {
        (*normalSel)[0 * 4 + i].set(0.0, isqrtwo, isqrtwo);
        (*normalSel)[1 * 4 + i].set(isqrtwo, 0.0, isqrtwo);
        (*normalSel)[2 * 4 + i].set(0.0, -isqrtwo, isqrtwo);
        (*normalSel)[3 * 4 + i].set(-isqrtwo, 0.0, isqrtwo);
        (*normalSel)[4 * 4 + i].set(-isqrtwo, 0.0, isqrtwo);
        (*normalSel)[5 * 4 + i].set(isqrtwo, 0.0, isqrtwo);
    }

    verticesArraySel[0] = 0;
    verticesArraySel[1] = 4;
    verticesArraySel[2] = 5;
    verticesArraySel[3] = 1;
    verticesArraySel[4] = 1;
    verticesArraySel[5] = 5;
    verticesArraySel[6] = 6;
    verticesArraySel[7] = 2;
    verticesArraySel[8] = 3;
    verticesArraySel[9] = 7;
    verticesArraySel[10] = 4;
    verticesArraySel[11] = 0;
    verticesArraySel[12] = 4;
    verticesArraySel[13] = 8;
    verticesArraySel[14] = 9;
    verticesArraySel[15] = 5;
    verticesArraySel[16] = 5;
    verticesArraySel[17] = 9;
    verticesArraySel[18] = 10;
    verticesArraySel[19] = 6;
    verticesArraySel[20] = 7;
    verticesArraySel[21] = 11;
    verticesArraySel[22] = 8;
    verticesArraySel[23] = 4;

    vertices = new DrawElementsUShort(PrimitiveSet::QUADS, 32, verticesArray);
    verticesSel = new DrawElementsUShort(PrimitiveSet::QUADS, 24, verticesArraySel);

    delete[] verticesArray;
    delete[] verticesArraySel;

    textureMat = new osg::Material();
    textureMat->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    textureMat->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.2, 0.2, 0.2, 1.0));
    textureMat->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    textureMat->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    textureMat->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, 1.0));
    textureMat->setShininess(osg::Material::FRONT_AND_BACK, 80.0f);

    ref_ptr<osg::Material> mtl = new osg::Material();

    mtl->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
    mtl->setAmbient(osg::Material::FRONT_AND_BACK, Vec4(0.1, 0.1, 0.1, 1.0));
    mtl->setDiffuse(osg::Material::FRONT_AND_BACK, Vec4(0.6, 0.6, 0.6, 1.0));
    mtl->setSpecular(osg::Material::FRONT_AND_BACK, Vec4(1.0, 1.0, 1.0, 1.0));
    mtl->setEmission(osg::Material::FRONT_AND_BACK, Vec4(0.0, 0.0, 0.0, 1.0));
    mtl->setShininess(osg::Material::FRONT_AND_BACK, 80.0f);

    normalGeostate = new StateSet();
    normalGeostate->setGlobalDefaults();

    normalGeostate->setAttributeAndModes(OSGVruiPresets::getCullFaceBack(), StateAttribute::ON);
    normalGeostate->setAttributeAndModes(mtl.get(), StateAttribute::ON);
    normalGeostate->setMode(GL_BLEND, StateAttribute::ON);
    normalGeostate->setMode(GL_LIGHTING, StateAttribute::ON);
}

ref_ptr<Group> coPinEditor::createBackgroundGroup()
{

    ref_ptr<Geometry> geoset1 = new Geometry();
    ref_ptr<Geometry> geoset3 = new Geometry();
    ref_ptr<Geometry> geoset4 = new Geometry();
    ref_ptr<Geometry> geoset5 = new Geometry();

    ref_ptr<Geode> normalGeode = new Geode();
    ref_ptr<Geode> colorGeode = new Geode();
    backgroundGeode = new Geode();

    geoset1->setVertexArray(coord.get());
    geoset1->addPrimitiveSet(vertices.get());
    geoset1->setColorArray(color.get());
    geoset1->setColorBinding(Geometry::BIND_OVERALL);
    geoset1->setNormalArray(normal.get());
    geoset1->setNormalBinding(Geometry::BIND_PER_VERTEX);

    geoset3->setVertexArray(coordSel.get());
    geoset3->addPrimitiveSet(verticesSel.get());
    geoset3->setColorArray(color.get());
    geoset3->setColorBinding(Geometry::BIND_OVERALL);
    geoset3->setNormalArray(normalSel.get());
    geoset3->setNormalBinding(Geometry::BIND_PER_VERTEX);

    geoset4->setVertexArray(coordColor.get());
    geoset4->addPrimitiveSet(verticesSel.get());
    geoset4->setColorArray(color.get());
    geoset4->setColorBinding(Geometry::BIND_OVERALL);
    geoset4->setNormalArray(normalSel.get());
    geoset4->setNormalBinding(Geometry::BIND_PER_VERTEX);

    backgroundGeometry = new Geometry();
    backgroundGeometry->setColorArray(color.get());
    backgroundGeometry->setColorBinding(Geometry::BIND_OVERALL);
    backgroundGeometry->setTexCoordArray(0, texcoord.get());
    backgroundGeometry->setVertexArray(coordt.get());
    backgroundGeometry->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    backgroundGeometry->setNormalArray(normalt.get());
    backgroundGeometry->setNormalBinding(Geometry::BIND_OVERALL);

    geoset5->setColorArray(color.get());
    geoset5->setColorBinding(Geometry::BIND_OVERALL);
    geoset5->setTexCoordArray(0, texcoordColor.get());
    geoset5->setVertexArray(coordColort.get());
    geoset5->addPrimitiveSet(new DrawArrays(PrimitiveSet::QUADS, 0, 4));
    geoset5->setNormalArray(normalt.get());
    geoset5->setNormalBinding(Geometry::BIND_OVERALL);

    ref_ptr<StateSet> ColorGeostate = new StateSet();
    ColorGeostate->setGlobalDefaults();
    ColorGeostate->setAttributeAndModes(textureMat.get(), StateAttribute::ON);

    // create Texture Object
    tex = new Texture2D();
#ifndef __linux__
    tex->setFilter(Texture::MIN_FILTER, Texture::NEAREST);
#endif
    tex->setFilter(Texture::MAG_FILTER, Texture::NEAREST);
    tex->setWrap(Texture::WRAP_S, Texture::CLAMP);
    tex->setWrap(Texture::WRAP_T, Texture::CLAMP);

    textureData = new unsigned char[3 * 4 * TEXTURE_RES_COLOR];
    myTransFunc->makeColorBar(TEXTURE_RES_COLOR, textureData, myFunctionEditor->getMin(), myFunctionEditor->getMax(), true);

    ref_ptr<Image> image = new Image();
    image->setImage(TEXTURE_RES_COLOR, 2, 1, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE,
                    textureData, Image::NO_DELETE, 4);
    tex->setImage(image.get());

    ColorGeostate->setTextureAttributeAndModes(0, tex.get(), StateAttribute::ON);
    ColorGeostate->setTextureAttributeAndModes(0, OSGVruiPresets::getTexEnvModulate(), StateAttribute::ON);
    ColorGeostate->setAttributeAndModes(OSGVruiPresets::getCullFaceBack(), StateAttribute::ON);
    ColorGeostate->setAttributeAndModes(OSGVruiPresets::getPolyModeFill(), StateAttribute::ON);
    ColorGeostate->setMode(GL_LIGHTING, StateAttribute::ON);
    ColorGeostate->setMode(GL_BLEND, StateAttribute::ON);

    HistoBackgroundGeostate = new StateSet();
    HistoBackgroundGeostate->setGlobalDefaults();
    HistoBackgroundGeostate->setAttributeAndModes(textureMat.get(), StateAttribute::ON);

    // create Texture Object
    histoTex = new Texture2D();
#ifndef __linux__
    histoTex->setFilter(Texture::MIN_FILTER, Texture::NEAREST);
#endif
    histoTex->setFilter(Texture::MAG_FILTER, Texture::NEAREST);
    histoTex->setWrap(Texture::WRAP_S, Texture::CLAMP);
    histoTex->setWrap(Texture::WRAP_T, Texture::CLAMP);

    HistoBackgroundGeostate->setTextureAttributeAndModes(0, histoTex.get());
    HistoBackgroundGeostate->setTextureAttributeAndModes(0, OSGVruiPresets::getTexEnvModulate(), StateAttribute::ON);
    HistoBackgroundGeostate->setAttributeAndModes(OSGVruiPresets::getCullFaceBack(), StateAttribute::ON);
    HistoBackgroundGeostate->setAttributeAndModes(OSGVruiPresets::getPolyModeFill(), StateAttribute::ON);
    HistoBackgroundGeostate->setMode(GL_LIGHTING, StateAttribute::ON);
    HistoBackgroundGeostate->setMode(GL_BLEND, StateAttribute::OFF);

    NormalBackgroundGeostate = new StateSet();
    NormalBackgroundGeostate->setGlobalDefaults();
    NormalBackgroundGeostate->setAttributeAndModes(textureMat.get(), StateAttribute::ON);

    // create Texture Object
    ref_ptr<Texture2D> backgroundTex = new Texture2D;
    backgroundTex->setFilter(Texture::MIN_FILTER, Texture::LINEAR);
    backgroundTex->setWrap(Texture::WRAP_S, Texture::REPEAT);
    backgroundTex->setWrap(Texture::WRAP_T, Texture::REPEAT);

    const char *textureName = "panel";
    const char *name = NULL;
    std::string look = covise::coCoviseConfig::getEntry("COVER.LookAndFeel");
    if (!look.empty())
    {
        char *fn = new char[strlen(textureName) + strlen(look.c_str()) + 50];
        sprintf(fn, "share/covise/icons/%s/Volume/%s.rgb", look.c_str(), textureName);
        name = coVRFileManager::instance()->getName(fn);
        delete[] fn;
    }
    if (name == NULL)
    {
        char *fn = new char[strlen(textureName) + 50];
        sprintf(fn, "share/covise/icons/Volume/%s.rgb", textureName);
        name = coVRFileManager::instance()->getName(fn);
        delete[] fn;
    }
    //cerr << textureName << "loading Texture " << name << endl;
    if (name)
    {
        ref_ptr<Image> image = osgDB::readImageFile(name);
        backgroundTex->setImage(image.get());
        NormalBackgroundGeostate->setTextureAttributeAndModes(0, backgroundTex.get(), StateAttribute::ON);
    }

    NormalBackgroundGeostate->setAttributeAndModes(OSGVruiPresets::getCullFaceBack(), StateAttribute::ON);
    NormalBackgroundGeostate->setAttributeAndModes(OSGVruiPresets::getPolyModeFill(), StateAttribute::ON);
    NormalBackgroundGeostate->setMode(GL_LIGHTING, StateAttribute::ON);
    NormalBackgroundGeostate->setMode(GL_BLEND, StateAttribute::ON);

    backgroundGeode->setStateSet(NormalBackgroundGeostate.get());
    normalGeode->setStateSet(normalGeostate.get());
    colorGeode->setStateSet(ColorGeostate.get());

    backgroundGeode->addDrawable(backgroundGeometry.get());
    normalGeode->addDrawable(geoset1.get());
    normalGeode->addDrawable(geoset3.get());
    normalGeode->addDrawable(geoset4.get());
    colorGeode->addDrawable(geoset5.get());

    backgroundGroup = new Group();
    backgroundGroup->addChild(backgroundGeode.get());
    backgroundGroup->addChild(normalGeode.get());
    backgroundGroup->addChild(colorGeode.get());

    return backgroundGroup.get();
}

void coPinEditor::adjustSelectionBar()
{
    w1 = 0.0;
    w2 = 0.0;
    w3 = 0.0;
    w4 = 0.0;
    vvTFPyramid *pyrPin = dynamic_cast<vvTFPyramid *>(currentPin);
    vvTFSkip *skipPin = dynamic_cast<vvTFSkip *>(currentPin);
    if (pyrPin)
    {
        w1 = ((coAlphaHatPin *)currentPin)->w1;
        w2 = ((coAlphaHatPin *)currentPin)->w2;
        w3 = ((coAlphaHatPin *)currentPin)->w3;
    }
    else if (skipPin)
    {
        w1 = ((coAlphaBlankPin *)currentPin)->w1;
        w2 = ((coAlphaBlankPin *)currentPin)->w2;
        w3 = ((coAlphaBlankPin *)currentPin)->w3;
    }

    //cerr << "W1: " << w1 << " W2: " << w2 << " W3: " << w3 << endl;

    (*selectionBarCoord)[0].set(A + B, -(A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[1].set(A + B + W * w1, -(A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[2].set(A + B + W * (w1 + w2), -(A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[3].set(A + B + W * (w1 + w2 + w3), -(A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[4].set(A + B + W * (w1 + w2 + w3 + w4), -(A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[5].set(A + B + W * (w1 + w2 + w3 + w4), -(SELH + A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[6].set(A + B + W * (w1 + w2 + w3), -(SELH + A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[7].set(A + B + W * (w1 + w2), -(SELH + A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[8].set(A + B + W * w1, -(SELH + A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[9].set(A + B, -(SELH + A + B + COLORH + A + B), ZOFFSET);

    selectionBarGeometry->dirtyDisplayList();
    selectionBarCoord->dirty();
    selectionBarGeode->dirtyBound();
}

void coPinEditor::createSelectionBarLists()
{

    selectionBarColor = new Vec4Array(16);
    selectionBarCoord = new Vec3Array(10);
    selectionBarNormal = new Vec3Array(1);

    ushort *selectionBarVerticesArray = new ushort[4 * 4];

    (*selectionBarCoord)[0].set(A + B, -(A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[1].set(A + B + W * 0.2, -(A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[2].set(A + B + W * 0.5, -(A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[3].set(A + B + W * 0.8, -(A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[4].set(A + B + W, -(A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[5].set(A + B + W, -(SELH + A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[6].set(A + B + W * 0.8, -(SELH + A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[7].set(A + B + W * 0.5, -(SELH + A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[8].set(A + B + W * 0.2, -(SELH + A + B + COLORH + A + B), ZOFFSET);
    (*selectionBarCoord)[9].set(A + B + 0.0, -(SELH + A + B + COLORH + A + B), ZOFFSET);

    for (int i = 0; i < 4; ++i)
    {
        (*selectionBarColor)[0 * 4 + i].set(0.2f, 0.2f, 0.2f, 1.0f);
        (*selectionBarColor)[1 * 4 + i].set(0.5f, 0.5f, 0.5f, 1.0f);
        (*selectionBarColor)[2 * 4 + i].set(0.2f, 0.2f, 0.2f, 1.0f);
        (*selectionBarColor)[3 * 4 + i].set(0.5f, 0.5f, 0.5f, 1.0f);
    }

    (*selectionBarNormal)[0].set(0.0, 0.0, 1.0);

    selectionBarVerticesArray[0] = 9;
    selectionBarVerticesArray[1] = 8;
    selectionBarVerticesArray[2] = 1;
    selectionBarVerticesArray[3] = 0;

    selectionBarVerticesArray[4] = 8;
    selectionBarVerticesArray[5] = 7;
    selectionBarVerticesArray[6] = 2;
    selectionBarVerticesArray[7] = 1;

    selectionBarVerticesArray[8] = 7;
    selectionBarVerticesArray[9] = 6;
    selectionBarVerticesArray[10] = 3;
    selectionBarVerticesArray[11] = 2;

    selectionBarVerticesArray[12] = 6;
    selectionBarVerticesArray[13] = 5;
    selectionBarVerticesArray[14] = 4;
    selectionBarVerticesArray[15] = 3;

    selectionBarVertices = new DrawElementsUShort(PrimitiveSet::QUADS, 16, selectionBarVerticesArray);

    delete[] selectionBarVerticesArray;
}

ref_ptr<Geode> coPinEditor::createSelectionBarGeode()
{

    selectionBarGeometry = new Geometry();

    selectionBarGeometry->setColorArray(selectionBarColor.get());
    selectionBarGeometry->setColorBinding(Geometry::BIND_PER_VERTEX);
    selectionBarGeometry->setVertexArray(selectionBarCoord.get());
    selectionBarGeometry->addPrimitiveSet(selectionBarVertices.get());
    selectionBarGeometry->setNormalArray(selectionBarNormal.get());
    selectionBarGeometry->setNormalBinding(Geometry::BIND_OVERALL);

    selectionBarGeode = new Geode();
    selectionBarGeode->setStateSet(normalGeostate.get());
    selectionBarGeode->addDrawable(selectionBarGeometry.get());
    return selectionBarGeode.get();
}

coPin *coPinEditor::findPin(int context)
{
    coPin *myPin = NULL;
    for (list<coPin *>::iterator pin = pinList.begin(); pin != pinList.end(); ++pin)
    {
        if ((*pin)->getID() == context)
        {
            myPin = (*pin);
            break;
        }
    }
    return myPin;
}
