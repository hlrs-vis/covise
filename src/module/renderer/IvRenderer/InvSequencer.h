/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_SEQUENCER_
#define _INV_SEQUENCER_

/* $Log:  $
 * Revision 1.1  1994/07/17  13:39:31  zrfu0390
 * Initial revision
 * */
/*
#ifndef  SYNC_LOOSE
#define SYNC_LOOSE 1
#endif
#ifndef  SYNC_SYNC
#define SYNC_SYNC 1
#endif
#ifndef  SYNC_TIGHT
#define SYNC_TIGHT 2
#endif*/

// **************************************************************************
//
// * Description    : Inventor Sequencer
//
// * Class(es)      : InvSequencer
//
// * inherited from : SoXtComponent
//
// * Author  : Dirk Rantzau
//
// * History : 17.07.95 V 1.0
//
// **************************************************************************

#include <covise/covise.h>
#include <X11/Intrinsic.h>
#include <Xm/Xm.h>
#include <Inventor/Xt/SoXtComponent.h>
#include <Inventor/misc/SoCallbackList.h>
#include <Inventor/sensors/SoSensor.h>

typedef enum
{
    INACTIVE,
    ACTIVE
} activeState;
typedef enum
{
    STOP,
    FORWARD,
    BACKWARD,
    PLAYFORWARD,
    PLAYBACKWARD
} playState;
class InvSequencer;

// callback function prototypes
typedef void InvSequencerCB(void *userData, InvSequencer *seq);
typedef void SequencerCallback(void *userData, void *callbackData);

//============================================================================
//
//  Class: InvSequencer
//
//  Sequencer for time frame stepping
//
//============================================================================
class InvSequencer : public SoXtComponent
{
public:
    InvSequencer(Widget parent = NULL,
                 const char *name = NULL,
                 SbBool buildInsideParent = TRUE,
                 SbBool showMaterialName = FALSE);
    ~InvSequencer();

    void setSliderBounds(int min, int max, int val, int slidermin, int slidermax);
    int getValue()
    {
        return val;
    }
    int getMinimum()
    {
        return min;
    }
    int getMaximum()
    {
        return max;
    }
    int getSliderMinimum()
    {
        return boundmin;
    }
    int getSliderMaximum()
    {
        return boundmax;
    }

    int getSeqState()
    {
        return seqState;
    };
    void setSeqState(const int &s)
    {
        seqState = s;
    };
    int getOldState()
    {
        return oldState_;
    };

    void setValueChangedCallback(SequencerCallback *func, void *userData);
    void removeValueChangedCallback();
    void setActive();
    void setInactive();
    void setSyncMode(char *message, int isMaster);
    void snap();
    void show();
    void hide();
    void stop(); // explicit call to stop the sequencer
    void play(int state); // explicit call to start "PLAY" on the sequencer
    void activate()
    {
        seqActive = ACTIVE;
    };
    int getSeqAct()
    {
        return seqActive;
    };
    void setSeqAct(const int &s)
    {
        seqActive = s;
    };
    int noSeq;

    int shown()
    {
        return shown_;
    };

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call InvObjectEditor::buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    InvSequencer(Widget parent,
                 const char *name,
                 SbBool buildInsideParent,
                 SbBool showMaterialName,
                 SbBool buildNow);

    // redefine these
    virtual const char *getDefaultWidgetName() const;
    virtual const char *getDefaultTitle() const;
    virtual const char *getDefaultIconTitle() const;

    // build routines
    Widget buildWidget(Widget parent);

private:
    int shown_;
    Dimension width_;

    SequencerCallback *valueChangedCallback;
    void *valueChangedUserData;
    void *valueChangedCallbackData;
    char MinBuffer[10], MaxBuffer[10], ValBuffer[10];
    int min, boundmin, max, boundmax, val; // slider state
    Widget form, part1, part2, label[5], labelmin, labelmax, slider;
    Pixmap pix_records[5]; // pixmap with player symbols
    Pixel bg, fg;
    void definePixmaps(Widget parent);
    static void inputCB(Widget w, int num, XmAnyCallbackStruct *);
    static void sliderCB(Widget w, void *userData, XmAnyCallbackStruct *);
    static void sequencerCB(Widget w, int num, XmPushButtonCallbackStruct *);
    void callValueChangedCallback(void *valueChangedUserData, void *valueChangedCallbackData)
    {
        (*valueChangedCallback)(valueChangedUserData, valueChangedCallbackData);
    }
    // this is called by both constructors
    void constructorCommon(SbBool showMaterialName, SbBool buildNow);
    int sync_flag;
    int seqActive;
    int seqState;
    int oldState_;
    SoSensor *animSensor;
    static void AnimCB(void *, SoSensor *);
    void thisAnimCB(SoSensor *);
    void saveState()
    {
        oldState_ = seqState;
    };
};
#endif // _INV_SEQUENCER_
