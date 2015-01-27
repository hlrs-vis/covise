/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _XENOMAI_PLUGIN_H
#define _XENOMAI_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: Xenomai Plugin (does nothing)                              **
 **                                                                          **
 **                                                                          **
 ** Author: F.Seybold, S. Franz		                                                **
 **                                                                          **
 ** History:  								                                **
 ** Nov-01  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>

//#include "LinearMotorControlTask.h"
//#include "CanOpenDevice.h"
#include "ValidateMotionPlatform.h"
//#include "CanOpenController.h"
//#include "XenomaiSteeringWheel.h"
//#include "GasPedalControlTask.h"
//#include "BrakePedal.h"

using namespace opencover;
using namespace covise;

class XenomaiPlugin : public coVRPlugin, public coTUIListener
{
public:
    XenomaiPlugin();
    ~XenomaiPlugin();

    //initialization
    bool init();

    // this will be called in PreFrame
    void preFrame();

    // this will be called if an object with feedback arrives
    void newInteractor(RenderObject *container, coInteractor *i);

    // this will be called if a COVISE object arrives
    void addObject(RenderObject *container,
                   RenderObject *obj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   osg::Group *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn, float transparency);

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace);

private:
    coTUITab *xenoTab;

    void tabletPressEvent(coTUIElement *tUIItem);

    //XenomaiSocketCan can0;
    //CanOpenController con0;

    //LinearMotorControlTask linMot;
    ValidateMotionPlatform *motPlat;

    coTUILabel *linMotLabel;
    coTUILabel *posLabel;
    coTUILabel *velLabel;
    coTUILabel *accLabel;
    coTUILabel *retLabel;
    coTUILabel *overrunLabel;

    coTUILabel *linMotOneLabel;
    coTUISlider *linMotOnePosSlider;
    coTUISlider *linMotOneVelSlider;
    coTUISlider *linMotOneAccSlider;
    coTUILabel *linMotTwoLabel;
    coTUISlider *linMotTwoPosSlider;
    coTUISlider *linMotTwoVelSlider;
    coTUISlider *linMotTwoAccSlider;
    coTUILabel *linMotThreeLabel;
    coTUISlider *linMotThreePosSlider;
    coTUISlider *linMotThreeVelSlider;
    coTUISlider *linMotThreeAccSlider;

    coTUILabel *rotMotOneLabel;
    coTUISlider *rotMotOneTorqueSlider;

    coTUIButton *linMotOnePositionierungButton;
    coTUIButton *linMotOneIntPositionierungButton;
    coTUIButton *linMotOneEncoderButton;
    coTUIButton *linMotOneResetButton;
    coTUIButton *linMotOneEndstufesperreButton;
    coTUIButton *linMotTwoPositionierungButton;
    coTUIButton *linMotTwoIntPositionierungButton;
    coTUIButton *linMotTwoEncoderButton;
    coTUIButton *linMotTwoResetButton;
    coTUIButton *linMotTwoEndstufesperreButton;
    coTUIButton *linMotThreePositionierungButton;
    coTUIButton *linMotThreeIntPositionierungButton;
    coTUIButton *linMotThreeEncoderButton;
    coTUIButton *linMotThreeResetButton;
    coTUIButton *linMotThreeEndstufesperreButton;

    coTUIButton *rotMotOneTorqueButton;
    coTUIButton *rotMotOneReferenceButton;
    coTUIButton *rotMotOneResetButton;
    coTUIButton *rotMotOneEndstufesperreButton;

    coTUILabel *linMotOneAnswerLabel;
    coTUILabel *linMotTwoAnswerLabel;
    coTUILabel *linMotThreeAnswerLabel;
    coTUILabel *rotMotOneAnswerLabel;

    //CanOpenController con1;
    //XenomaiSteeringWheel steeringWheel;

    coTUILabel *steeringWheelLabel;
    coTUIButton *steeringWheelHomingButton;
    coTUIButton *steeringWheelStart;
    coTUILabel *steeringWheelTaskOverrunsLabel;
    coTUISlider *rumbleAmplitudeSlider;

    //XenomaiSocketCan can3;
    //GasPedalControlTask pedalTask;

    coTUILabel *gasPedalLabel;
    coTUILabel *gasPedalPosition;

    //CanOpenController con3;
    //BrakePedal brakePedalTask;

    coTUILabel *brakePedalLabel;
    coTUILabel *brakePedalPosition;
};
#endif
