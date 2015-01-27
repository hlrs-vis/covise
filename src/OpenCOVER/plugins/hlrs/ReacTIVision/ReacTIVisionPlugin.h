/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ReacTIVision_PLUGIN_H
#define _ReacTIVision_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2009 HLRS  **
**                                                                          **
** Description: ReacTIVision Plugin (does also TouchTable)                  **
**                                                                          **
**                                                                          **
** Author: B. Burbaum                                                       **
**                		                                                    **
**                                                                          **
** History:  								                                **
** Feb-09  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
//#include <ITouchScreen.h>
//#include <ITouchListener.h>

#include "TuioListener.h"
#include "TuioClient.h"
#include "coVRTouchTable.h"

//     #include <math.h>
//   #include <OpenThreads/Mutex>   //zieht nach client.h

class ReacTIVisionPlugin : public coVRPlugin, public coTUIListener, public coVRTouchTableInterface // ,    public TuioListener

//class ReacTIVisionPlugin : public coVRPlugin, public touchlib::ITouchListener, public coTUIListener

{
public:
    ReacTIVisionPlugin();
    ~ReacTIVisionPlugin();

    TuioClient *client; // wichtig

    //	OpenThreads::Mutex mutex;           //zieht nach client.h
    //	bool hasChanged;

    // this will be called in PreFrame
    virtual void preFrame();

    // for class COVEREXPORT coVRTouchTableInterface
    virtual int getMarker(std::string name);
    virtual bool isVisible(int);
    virtual osg::Vec2 getPosition(int);
    virtual float getOrientation(int);

private:
    //	touchlib::ITouchScreen *screen;

    /* 	coTUITab *ReacTIVisionTab;
	coTUIToggleButton *changeFingerSize;
	coTUIToggleButton *NoShowButton;
	coTUIToggleButton *recapture_background;
	coTUIFloatSlider  *FingerSizeSlider;
	coTUILabel *MyFirstLabel;

#define  MaxFilter  10
#define  MaxSlider  5

	std::vector<coTUIToggleButton*> myButtons;
	//coTUIToggleButton *MyButton;
	coTUIFloatSlider*  FilterSlider[MaxFilter][MaxSlider];
	bool isNoVisible[MaxFilter];

	// bool isThresholdValue;
	//int ThresholdValue;

*/
    float FingerSizeValue;
    int buttonState;
};
#endif
