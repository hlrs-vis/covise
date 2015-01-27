/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TouchTable_PLUGIN_H
#define _TouchTable_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: TouchTable Plugin (does TouchTable)                         **
**                                                                          **
**                                                                          **
** Author: B. Burbaum und                                                  **
**         U. Woessner		                                               **
**                                                                          **
** History:  								                                **
** Nov-01  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <ITouchScreen.h>
#include <ITouchListener.h>
#include <TouchData.h>

class TouchTablePlugin : public coVRPlugin, public touchlib::ITouchListener, public coTUIListener
{
public:
    TouchTablePlugin();
    ~TouchTablePlugin();

    // this will be called in PreFrame
    void preFrame();

    void key(int type, int keySym, int mod);

    //! Notify that a finger has just been made active.
    virtual void fingerDown(TouchData data);

    //! Notify that a finger has just been made active.
    virtual void fingerUpdate(TouchData data);

    //! A finger is no longer active..
    virtual void fingerUp(TouchData data);

    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

private:
    coTUITab *TouchTableTab;
    touchlib::ITouchScreen *screen;
    coTUIToggleButton *changeFingerSize;
    coTUIToggleButton *NoShowButton;
    coTUIToggleButton *recapture_background;
    coTUIFloatSlider *FingerSizeSlider;

    coTUILabel *MyFirstLabel;

#define MaxFilter 10
#define MaxSlider 5

    std::vector<coTUIToggleButton *> myButtons;
    //coTUIToggleButton *MyButton;
    coTUIFloatSlider *FilterSlider[MaxFilter][MaxSlider];
    bool isNoVisible[MaxFilter];

    bool isVisible;
    bool isNixShow;
    bool isRecapBGR;
    // bool isThresholdValue;
    //int ThresholdValue;
    float FingerSizeValue;
    int buttonState;
};
#endif
