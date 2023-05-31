/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Xenomai Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: F.Seybold, , S. Franz	                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "XenomaiPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>

#include <sstream>

XenomaiPlugin::XenomaiPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
    , //con0("rtcan0"),
    //can1("rtcan1"),
    //linMot(con0),
    motPlat(ValidateMotionPlatform::instance())
//con1("rtcan1"),
//steeringWheel(con1, 1),
//can3("rtcan3"),
//pedalTask(can3)
//con3("rtcan3"),
//brakePedalTask(con3, 21)
{
    fprintf(stderr, "XenomaiPlugin::XenomaiPlugin\n");
}

// this is called if the plugin is removed at runtime
XenomaiPlugin::~XenomaiPlugin()
{
    fprintf(stderr, "XenomaiPlugin::~XenomaiPlugin\n");

    delete xenoTab;
}

void
XenomaiPlugin::preFrame()
{
    /*linMot.setPositionOne(linMotOnePosSlider->getValue());
   linMot.setPositionTwo(linMotTwoPosSlider->getValue());
   linMot.setPositionThree(linMotThreePosSlider->getValue());
   linMot.setVelocityOne(linMotOneVelSlider->getValue());
   linMot.setVelocityTwo(linMotTwoVelSlider->getValue());
   linMot.setVelocityThree(linMotThreeVelSlider->getValue());
   linMot.setAccelerationOne(linMotOneAccSlider->getValue());
   linMot.setAccelerationTwo(linMotTwoAccSlider->getValue());
   linMot.setAccelerationThree(linMotThreeAccSlider->getValue());*/
    motPlat->getSendMutex().acquire();
    motPlat->setPositionSetpoint(0, (double)linMotOnePosSlider->getValue() / 1000);
    motPlat->setPositionSetpoint(1, (double)linMotTwoPosSlider->getValue() / 1000);
    motPlat->setPositionSetpoint(2, (double)linMotThreePosSlider->getValue() / 1000);
    motPlat->setVelocitySetpoint(0, (double)linMotOneVelSlider->getValue() / 1000);
    motPlat->setVelocitySetpoint(1, (double)linMotTwoVelSlider->getValue() / 1000);
    motPlat->setVelocitySetpoint(2, (double)linMotThreeVelSlider->getValue() / 1000);
    motPlat->setAccelerationSetpoint(0, (double)linMotOneAccSlider->getValue() / 1000);
    motPlat->setAccelerationSetpoint(1, (double)linMotTwoAccSlider->getValue() / 1000);
    motPlat->setAccelerationSetpoint(2, (double)linMotThreeAccSlider->getValue() / 1000);
    motPlat->getSendMutex().release();

    motPlat->setBrakePedalForce((double)rotMotOneTorqueSlider->getValue());

    /*std::stringstream answerOneStream;
   answerOneStream << linMot.getPositionOne() << ", " << linMot.getVelocityOne();
   linMotOneAnswerLabel->setLabel(answerOneStream.str().c_str());
   std::stringstream answerTwoStream;
   answerTwoStream << linMot.getPositionTwo() << ", " << linMot.getVelocityTwo();
   linMotTwoAnswerLabel->setLabel(answerTwoStream.str().c_str());
   std::stringstream answerThreeStream;
   answerThreeStream << linMot.getPositionThree() << ", " << linMot.getVelocityThree();
   linMotThreeAnswerLabel->setLabel(answerThreeStream.str().c_str());

   std::stringstream overrunStream;
   overrunStream << linMot.getPeriodicTaskOverruns();
   overrunLabel->setLabel(overrunStream.str().c_str());*/

    std::stringstream stateOneStream;
    stateOneStream << motPlat->getPosition(0) << ", " << motPlat->getVelocity(0);
    linMotOneAnswerLabel->setLabel(stateOneStream.str().c_str());
    std::stringstream stateTwoStream;
    stateTwoStream << motPlat->getPosition(1) << ", " << motPlat->getVelocity(1);
    linMotTwoAnswerLabel->setLabel(stateTwoStream.str().c_str());
    std::stringstream stateThreeStream;
    stateThreeStream << motPlat->getPosition(2) << ", " << motPlat->getVelocity(2);
    linMotThreeAnswerLabel->setLabel(stateThreeStream.str().c_str());

    std::stringstream stateRotMotOneStream;
    stateRotMotOneStream << motPlat->getBrakePedalPosition() << ", " << motPlat->getBrakePedalVelocity();
    rotMotOneAnswerLabel->setLabel(stateRotMotOneStream.str().c_str());

    /*std::stringstream labelStream;
   labelStream << steeringWheel.getPeriodicTaskOverruns();
   steeringWheelTaskOverrunsLabel->setLabel(labelStream.str().c_str());
   steeringWheel.setRumbleAmplitude(rumbleAmplitudeSlider->getValue());

   std::stringstream pedalLabelStream;
   pedalLabelStream << pedalTask.getActualPositionValue();
   gasPedalPosition->setLabel(pedalLabelStream.str().c_str());

   std::stringstream brakePedalLabelStream;
   brakePedalLabelStream << (int)brakePedalTask.getPosition();
   brakePedalPosition->setLabel(brakePedalLabelStream.str().c_str());*/
}

bool XenomaiPlugin::init()
{
    xenoTab = new coTUITab("Xenomai", coVRTui::instance()->mainFolder->getID());
    xenoTab->setPos(0, 0);

    linMotLabel = new coTUILabel("Motor", xenoTab->getID());
    linMotLabel->setPos(0, 0);
    posLabel = new coTUILabel("Position [mm, N]:", xenoTab->getID());
    posLabel->setPos(1, 0);
    velLabel = new coTUILabel("Velocity [mm/s]:", xenoTab->getID());
    velLabel->setPos(3, 0);
    accLabel = new coTUILabel("Acceleration [mm/s^2]:", xenoTab->getID());
    accLabel->setPos(5, 0);
    retLabel = new coTUILabel("Answer:", xenoTab->getID());
    retLabel->setPos(8, 0);
    overrunLabel = new coTUILabel("Overruns", xenoTab->getID());
    overrunLabel->setPos(9, 0);

    linMotOneLabel = new coTUILabel("1:", xenoTab->getID());
    linMotOneLabel->setPos(0, 1);
    linMotOnePosSlider = new coTUISlider("target position 1", xenoTab->getID());
    //linMotOnePosSlider->setRange(linMot.posLowerBound, linMot.posUpperBound);
    linMotOnePosSlider->setRange((int)(motPlat->posMin * 1000), (int)(motPlat->posMax * 1000));
    linMotOnePosSlider->setValue(0);
    linMotOnePosSlider->setPos(1, 1);
    linMotOneVelSlider = new coTUISlider("target vel 1", xenoTab->getID());
    //linMotOneVelSlider->setRange(linMot.velLowerBound, linMot.velUpperBound);
    linMotOneVelSlider->setRange((int)(motPlat->velMin * 1000), (int)(motPlat->velMax * 1000));
    linMotOneVelSlider->setPos(3, 1);
    linMotOneAccSlider = new coTUISlider("target acceleration 1", xenoTab->getID());
    //linMotOneAccSlider->setRange(linMot.accLowerBound, linMot.accUpperBound);
    linMotOneAccSlider->setRange((int)(motPlat->accMin * 1000), (int)(motPlat->accMax * 1000));
    linMotOneAccSlider->setPos(5, 1);
    linMotOnePositionierungButton = new coTUIButton("Positionierung", xenoTab->getID());
    linMotOnePositionierungButton->setPos(1, 2);
    linMotOnePositionierungButton->setEventListener(this);
    linMotOneIntPositionierungButton = new coTUIButton("Int. Positionierung", xenoTab->getID());
    linMotOneIntPositionierungButton->setPos(2, 2);
    linMotOneIntPositionierungButton->setEventListener(this);
    linMotOneEncoderButton = new coTUIButton("Encoder", xenoTab->getID());
    linMotOneEncoderButton->setPos(3, 2);
    linMotOneEncoderButton->setEventListener(this);
    linMotOneResetButton = new coTUIButton("Reset", xenoTab->getID());
    linMotOneResetButton->setPos(4, 2);
    linMotOneResetButton->setEventListener(this);
    linMotOneEndstufesperreButton = new coTUIButton("Endstufe sperren", xenoTab->getID());
    linMotOneEndstufesperreButton->setPos(5, 2);
    linMotOneEndstufesperreButton->setEventListener(this);
    linMotOneAnswerLabel = new coTUILabel("answer one", xenoTab->getID());
    linMotOneAnswerLabel->setPos(6, 2);

    linMotTwoLabel = new coTUILabel("2:", xenoTab->getID());
    linMotTwoLabel->setPos(0, 3);
    linMotTwoPosSlider = new coTUISlider("target position 2", xenoTab->getID());
    //linMotTwoPosSlider->setRange(linMot.posLowerBound, linMot.posUpperBound);
    linMotTwoPosSlider->setRange((int)(motPlat->posMin * 1000), (int)(motPlat->posMax * 1000));
    linMotTwoPosSlider->setValue(0);
    linMotTwoPosSlider->setPos(1, 3);
    linMotTwoVelSlider = new coTUISlider("target vel 1", xenoTab->getID());
    //linMotTwoVelSlider->setRange(linMot.velLowerBound, linMot.velUpperBound);
    linMotTwoVelSlider->setRange((int)(motPlat->velMin * 1000), (int)(motPlat->velMax * 1000));
    linMotTwoVelSlider->setPos(3, 3);
    linMotTwoAccSlider = new coTUISlider("target acceleration 2", xenoTab->getID());
    //linMotTwoAccSlider->setRange(linMot.accLowerBound, linMot.accUpperBound);
    linMotTwoAccSlider->setRange((int)(motPlat->accMin * 1000), (int)(motPlat->accMax * 1000));
    linMotTwoAccSlider->setPos(5, 3);
    linMotTwoPositionierungButton = new coTUIButton("Positionierung", xenoTab->getID());
    linMotTwoPositionierungButton->setPos(1, 4);
    linMotTwoPositionierungButton->setEventListener(this);
    linMotTwoIntPositionierungButton = new coTUIButton("Int. Positionierung", xenoTab->getID());
    linMotTwoIntPositionierungButton->setPos(2, 4);
    linMotTwoIntPositionierungButton->setEventListener(this);
    linMotTwoEncoderButton = new coTUIButton("Encoder", xenoTab->getID());
    linMotTwoEncoderButton->setPos(3, 4);
    linMotTwoEncoderButton->setEventListener(this);
    linMotTwoResetButton = new coTUIButton("Reset", xenoTab->getID());
    linMotTwoResetButton->setPos(4, 4);
    linMotTwoResetButton->setEventListener(this);
    linMotTwoEndstufesperreButton = new coTUIButton("Endstufe sperren", xenoTab->getID());
    linMotTwoEndstufesperreButton->setPos(5, 4);
    linMotTwoEndstufesperreButton->setEventListener(this);
    linMotTwoAnswerLabel = new coTUILabel("answer two", xenoTab->getID());
    linMotTwoAnswerLabel->setPos(6, 4);

    linMotThreeLabel = new coTUILabel("3:", xenoTab->getID());
    linMotThreeLabel->setPos(0, 5);
    linMotThreePosSlider = new coTUISlider("target position 3", xenoTab->getID());
    //linMotThreePosSlider->setRange(linMot.posLowerBound, linMot.posUpperBound);
    linMotThreePosSlider->setRange((int)(motPlat->posMin * 1000), (int)(motPlat->posMax * 1000));
    linMotThreePosSlider->setValue(0);
    linMotThreePosSlider->setPos(1, 5);
    linMotThreeVelSlider = new coTUISlider("target vel 1", xenoTab->getID());
    //linMotThreeVelSlider->setRange(linMot.velLowerBound, linMot.velUpperBound);
    linMotThreeVelSlider->setRange((int)(motPlat->velMin * 1000), (int)(motPlat->velMax * 1000));
    linMotThreeVelSlider->setPos(3, 5);
    linMotThreeAccSlider = new coTUISlider("target acceleration 3", xenoTab->getID());
    //linMotThreeAccSlider->setRange(linMot.accLowerBound, linMot.accUpperBound);
    linMotThreeAccSlider->setRange((int)(motPlat->accMin * 1000), (int)(motPlat->accMax * 1000));
    linMotThreeAccSlider->setPos(5, 5);
    linMotThreePositionierungButton = new coTUIButton("Positionierung", xenoTab->getID());
    linMotThreePositionierungButton->setPos(1, 6);
    linMotThreePositionierungButton->setEventListener(this);
    linMotThreeIntPositionierungButton = new coTUIButton("Int. Positionierung", xenoTab->getID());
    linMotThreeIntPositionierungButton->setPos(2, 6);
    linMotThreeIntPositionierungButton->setEventListener(this);
    linMotThreeEncoderButton = new coTUIButton("Nix", xenoTab->getID());
    linMotThreeEncoderButton->setPos(3, 6);
    linMotThreeEncoderButton->setEventListener(this);
    linMotThreeResetButton = new coTUIButton("Reset", xenoTab->getID());
    linMotThreeResetButton->setPos(4, 6);
    linMotThreeResetButton->setEventListener(this);
    linMotThreeEndstufesperreButton = new coTUIButton("Endstufe sperren", xenoTab->getID());
    linMotThreeEndstufesperreButton->setPos(5, 6);
    linMotThreeEndstufesperreButton->setEventListener(this);
    linMotThreeAnswerLabel = new coTUILabel("answer three", xenoTab->getID());
    linMotThreeAnswerLabel->setPos(6, 6);

    rotMotOneLabel = new coTUILabel("4:", xenoTab->getID());
    rotMotOneLabel->setPos(0, 7);
    rotMotOneTorqueSlider = new coTUISlider("target torque 4", xenoTab->getID());
    rotMotOneTorqueSlider->setRange(0, 350);
    rotMotOneTorqueSlider->setValue(0);
    rotMotOneTorqueSlider->setPos(1, 7);
    rotMotOneTorqueButton = new coTUIButton("Momentvorgabe", xenoTab->getID());
    rotMotOneTorqueButton->setPos(1, 8);
    rotMotOneTorqueButton->setEventListener(this);
    rotMotOneReferenceButton = new coTUIButton("Referenzfahrt", xenoTab->getID());
    rotMotOneReferenceButton->setPos(2, 8);
    rotMotOneReferenceButton->setEventListener(this);
    rotMotOneResetButton = new coTUIButton("Reset", xenoTab->getID());
    rotMotOneResetButton->setPos(4, 8);
    rotMotOneResetButton->setEventListener(this);
    rotMotOneEndstufesperreButton = new coTUIButton("Endstufe sperren", xenoTab->getID());
    rotMotOneEndstufesperreButton->setPos(5, 8);
    rotMotOneEndstufesperreButton->setEventListener(this);
    rotMotOneAnswerLabel = new coTUILabel("answer three", xenoTab->getID());
    rotMotOneAnswerLabel->setPos(6, 8);

    //linMot.start();
    motPlat->start();

    steeringWheelLabel = new coTUILabel("XenomaiSteeringWheel: ", xenoTab->getID());
    steeringWheelLabel->setPos(0, 9);
    steeringWheelHomingButton = new coTUIButton(("Start homing"), xenoTab->getID());
    steeringWheelHomingButton->setPos(1, 9);
    steeringWheelHomingButton->setEventListener(this);
    steeringWheelStart = new coTUIButton(("Start"), xenoTab->getID());
    steeringWheelStart->setPos(2, 9);
    steeringWheelStart->setEventListener(this);
    steeringWheelTaskOverrunsLabel = new coTUILabel("Overruns: ", xenoTab->getID());
    steeringWheelTaskOverrunsLabel->setPos(3, 9);
    rumbleAmplitudeSlider = new coTUISlider("slider", xenoTab->getID());
    rumbleAmplitudeSlider->setPos(4, 9);
    rumbleAmplitudeSlider->setRange(0, 100);

    gasPedalLabel = new coTUILabel("Gas pedal: ", xenoTab->getID());
    gasPedalLabel->setPos(0, 10);
    gasPedalPosition = new coTUILabel("0", xenoTab->getID());
    gasPedalPosition->setPos(1, 10);

    /*pedalTask.setMinTargetForce(0);
   pedalTask.setMaxTargetForce(pedalTask.MaxTargetForce);
   pedalTask.setTargetPositionValue(pedalTask.MaxTargetPositionValue/2);
   pedalTask.setJitterAmplitude(5);
   pedalTask.setJitterSignalForm(pedalTask.DemandJitterSignalFormSawTooth);
   pedalTask.setJitterFrequency(2);
   pedalTask.unlock();
   pedalTask.start();*/

    brakePedalLabel = new coTUILabel("Brake pedal: ", xenoTab->getID());
    brakePedalLabel->setPos(0, 11);
    brakePedalPosition = new coTUILabel("0", xenoTab->getID());
    brakePedalPosition->setPos(2, 11);
    //brakePedalTask.start();

    return (true);
}

void XenomaiPlugin::tabletPressEvent(coTUIElement *button) // tUIItem)
{
    if (button == linMotOnePositionierungButton)
    {
        //linMot.changeStateOne(linMot.statePositioning);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlPositioning>(0);
        motPlat->getSendMutex().release();
    }
    else if (button == linMotOneIntPositionierungButton)
    {
        //linMot.changeStateOne(linMot.stateReferenceSet);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlInterpolatedPositioning>(0);
        motPlat->getSendMutex().release();
    }
    else if (button == linMotOneEncoderButton)
    {
    }
    else if (button == linMotOneResetButton)
    {
        //linMot.changeStateOne(linMot.stateReset);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlReset>(0);
        motPlat->getSendMutex().release();
    }
    else if (button == linMotOneEndstufesperreButton)
    {
        //linMot.changeStateOne(linMot.stateDisable);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>(0);
        motPlat->getSendMutex().release();
    }

    else if (button == linMotTwoPositionierungButton)
    {
        //linMot.changeStateTwo(linMot.statePositioning);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlPositioning>(1);
        motPlat->getSendMutex().release();
    }
    else if (button == linMotTwoIntPositionierungButton)
    {
        //linMot.changeStateTwo(linMot.stateReferenceSet);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlInterpolatedPositioning>(1);
        motPlat->getSendMutex().release();
    }
    else if (button == linMotTwoEncoderButton)
    {
    }
    else if (button == linMotTwoResetButton)
    {
        //linMot.changeStateTwo(linMot.stateReset);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlReset>(1);
        motPlat->getSendMutex().release();
    }
    else if (button == linMotTwoEndstufesperreButton)
    {
        //linMot.changeStateTwo(linMot.stateDisable);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>(1);
        motPlat->getSendMutex().release();
    }

    else if (button == linMotThreePositionierungButton)
    {
        //linMot.changeStateThree(linMot.statePositioning);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlPositioning>(2);
        motPlat->getSendMutex().release();
    }
    else if (button == linMotThreeIntPositionierungButton)
    {
        //linMot.changeStateThree(linMot.stateReferenceSet);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlInterpolatedPositioning>(2);
        motPlat->getSendMutex().release();
    }
    else if (button == linMotThreeEncoderButton)
    {
    }
    else if (button == linMotThreeResetButton)
    {
        //linMot.changeStateThree(linMot.stateReset);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlReset>(2);
        motPlat->getSendMutex().release();
    }
    else if (button == linMotThreeEndstufesperreButton)
    {
        //linMot.changeStateThree(linMot.stateDisable);
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>(2);
        motPlat->getSendMutex().release();
    }

    else if (button == rotMotOneTorqueButton)
    {
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlTorque>(ValidateMotionPlatform::brakeMot);
        motPlat->getSendMutex().release();
    }
    else if (button == rotMotOneReferenceButton)
    {
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlReference>(ValidateMotionPlatform::brakeMot);
        motPlat->getSendMutex().release();
    }
    else if (button == rotMotOneResetButton)
    {
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlReset>(ValidateMotionPlatform::brakeMot);
        motPlat->getSendMutex().release();
    }
    else if (button == rotMotOneEndstufesperreButton)
    {
        motPlat->getSendMutex().acquire();
        motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>(ValidateMotionPlatform::brakeMot);
        motPlat->getSendMutex().release();
    }

    /*else if(button == steeringWheelHomingButton) {
      steeringWheel.center();
   }
   else if(button == steeringWheelStart) {
      steeringWheel.start();
   }*/
}

COVERPLUGIN(XenomaiPlugin)
