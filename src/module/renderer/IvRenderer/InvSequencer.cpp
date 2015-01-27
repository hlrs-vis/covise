/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log:  $
 * Revision 1.1  1994/07/17  13:39:31  zrfu0390
 * Initial revision
 * */

//**************************************************************************
//
// * Description    : Inventor Object Editor
//
// * Class(es)      : InvSequencer
//
// * inherited from : SoXtComponent
//
// * Author  : Dirk Rantzau
//
// * History : 17.07.95 V 1.0
//
//**************************************************************************

#include <string.h>
#include <math.h>
#include <util/coFileUtil.h>

#include "InvCoviseViewer.h"
#include <Xm/Xm.h>
#include <Xm/BulletinB.h>
#include <Xm/CascadeB.h>
#include <Xm/CascadeBG.h>
#include <Xm/FileSB.h>
#include <Xm/Form.h>
#include <Xm/List.h>
#include <Xm/Label.h>
#include <Xm/FileSB.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/SeparatoG.h>
#include <Xm/Text.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>
#include <Xm/RowColumn.h>
#include <Xm/Scale.h>

#include <Inventor/Xt/SoXt.h>
#include <Inventor/SbLinear.h>
#include <Inventor/SoDB.h>
#include <Inventor/SoInput.h>
#include <Inventor/SoPath.h>
#include <Inventor/SoLists.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/actions/SoSearchAction.h>
#include <Inventor/Xt/SoXtClipboard.h>
#include <Inventor/Xt/SoXtRenderArea.h>
#include <Inventor/errors/SoDebugError.h>
#include <Inventor/sensors/SoTimerSensor.h>

#include "InvSequencer.h"

void closeWindowCB(void * /*userData*/, SoXtComponent *)
{

    // InvSequencer *seq = (InvSequencer *)userData;

    // cerr << "Sequencer should be closed but isn't :-)" << endl;
}

//**************************************************************************
// CLASS InvSequencer
//**************************************************************************
//=======================================================================
//
// Public constructor - build the widget right now
//
//=======================================================================
InvSequencer::InvSequencer(Widget parent,
                           const char *name,
                           SbBool buildInsideParent,
                           SbBool showName)
    : SoXtComponent(parent,
                    name,
                    buildInsideParent)

{
    // In this case, this component is what the app wants, so buildNow = TRUE
    constructorCommon(showName, TRUE);
    coviseViewer->addSequencer(this);
}

//=========================================================================
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
//
//=========================================================================
InvSequencer::InvSequencer(Widget parent,
                           const char *name,
                           SbBool buildInsideParent,
                           SbBool showName,
                           SbBool buildNow)
    : SoXtComponent(parent,
                    name,
                    buildInsideParent)

{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
    // as it was passed to us.
    constructorCommon(showName, buildNow);
    coviseViewer->addSequencer(this);
}

//=========================================================================
//
// Called by the constructors
//
// private
//
//=========================================================================
void
InvSequencer::constructorCommon(SbBool showName, SbBool buildNow)

{
    (void)showName;

    // init local vars
    setClassName("InvSequencer");
    noSeq = 1;
    seqActive = INACTIVE;
    oldState_ = STOP;

    valueChangedCallback = NULL;
    valueChangedUserData = NULL;
    valueChangedCallbackData = NULL; // not used

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }

    // default sync mode
    sync_flag = SYNC_TIGHT;

    // create an idle sensor for animation stuff, do not schedule for now
    SoTimerSensor *timer = new SoTimerSensor;
    //animSensor->setPriority(1000000);
    animSensor = timer;
    timer->setInterval(1. / 25.);
    timer->setFunction(InvSequencer::AnimCB);
    timer->setData((void *)this);
    timer->unschedule();

    // avoid being closed
    setWindowCloseCallback(closeWindowCB, this);
}

//=========================================================================
//
//    Destructor.
//
InvSequencer::~InvSequencer()
//
//=========================================================================
{
    if (animSensor->isScheduled())
        animSensor->unschedule();
    delete animSensor;
    coviseViewer->removeSequencer(this);

    // destroy all widgets of the sequencer
    XtDestroyWidget(labelmin);
    XtDestroyWidget(labelmax);
    XtDestroyWidget(slider);
    int i;
    for (i = 0; i < 5; ++i)
    {
        XtDestroyWidget(label[i]);
    }
}

//=========================================================================
//
// Description:
//  	Buils the editor layout
//
// Use: protected
//
//=========================================================================
Widget
InvSequencer::buildWidget(Widget parent)
{
    long i;

    shown_ = 1;

    definePixmaps(parent);

    // set default slider bounds;
    min = boundmin = 0;
    max = boundmax = 2;
    val = 1;
    strcpy(MinBuffer, "0");
    strcpy(ValBuffer, "1");
    strcpy(MaxBuffer, "2");

    form = parent;

    part2 = parent;

    // create the minimum text widget //
    labelmin = XtVaCreateManagedWidget("Text",
                                       xmTextWidgetClass, part2,
                                       XmNuserData, this,
                                       XmNcolumns, 4,
                                       XmNvalue, MinBuffer,
                                       XmNtopAttachment, XmATTACH_FORM,
                                       XmNtopOffset, 30,
                                       XmNleftAttachment, XmATTACH_FORM,
                                       //    XmNcursorPosition,  strlen(port->Item[0])+1,
                                       NULL);

    XtAddCallback(labelmin, XmNactivateCallback,
                  (XtCallbackProc)inputCB, (XtPointer)10);

    // create a slider widget //
    slider = XtVaCreateManagedWidget("Slider",
                                     xmScaleWidgetClass, part2,
                                     XmNuserData, this,
                                     XmNminimum, min,
                                     XmNmaximum, max,
                                     XmNvalue, val,
                                     XmNshowValue, TRUE,
                                     XmNorientation, XmHORIZONTAL,
                                     XmNscaleWidth, 150,
                                     XmNtopAttachment, XmATTACH_FORM,
                                     XmNtopOffset, 30,
                                     XmNleftAttachment, XmATTACH_WIDGET,
                                     XmNleftWidget, labelmin,
                                     XmNwidth, 150,
                                     XmNheight, 35,
                                     NULL);
    XtAddCallback(slider, XmNvalueChangedCallback,
                  (XtCallbackProc)sliderCB, (XtPointer) this);
    XtAddCallback(slider, XmNdragCallback,
                  (XtCallbackProc)sliderCB, (XtPointer) this);

    // create the maximum text widget //
    labelmax = XtVaCreateManagedWidget("Text",
                                       xmTextWidgetClass, part2,
                                       XmNuserData, this,
                                       XmNcolumns, 4,
                                       XmNvalue, MaxBuffer,
                                       XmNtopAttachment, XmATTACH_FORM,
                                       XmNtopOffset, 30,
                                       XmNleftAttachment, XmATTACH_WIDGET,
                                       XmNleftWidget, slider,
                                       XmNrightAttachment, XmATTACH_FORM,
                                       //    XmNcursorPosition,  strlen(port->Item[1])+1,
                                       NULL);
    XtAddCallback(labelmax, XmNactivateCallback,
                  (XtCallbackProc)inputCB, (XtPointer)11);

    XtManageChild(part2);

    // create the pixmap buttons //
    //px = 120;
    //py = 10;
    int xpos = 10;
    for (i = 0; i < 5; i++)
    {
        label[i] = XtVaCreateManagedWidget("Arrow",
                                           xmPushButtonWidgetClass, part2,
                                           XmNuserData, this,
                                           XmNlabelType, XmPIXMAP,
                                           XmNlabelPixmap, pix_records[i],
                                           //XmNx,		px,
                                           //XmNy,		py,
                                           XmNleftPosition, xpos,
                                           XmNleftAttachment, XmATTACH_POSITION,
                                           XmNtopAttachment, XmATTACH_WIDGET,
                                           XmNtopOffset, 10,
                                           XmNtopWidget, slider,
                                           NULL);
        //px = px + 45;
        xpos += 17;
        XtAddCallback(label[i], XmNactivateCallback,
                      (XtCallbackProc)sequencerCB, (XtPointer)(i - 2));
    }

    //     XtVaGetValues(part2,
    // 		  XmNwidth, &width_,
    // 		  NULL);

    width_ = 250;

    //XtManageChild(part1);
    // XtSetSensitive(part1, 1);
    XtManageChild(part2);
    XtVaSetValues(parent,
                  XmNresizeWidth, True,
                  NULL);

    // hack to overcome the linux problem
    hide();
    show();

    return form;
}

void
InvSequencer::show()
{

    XtVaSetValues(form,
                  XmNwidth, width_,
                  NULL);

    if ((seqActive == ACTIVE) && (noSeq == 1))
    {
        XtManageChild(form);
        shown_ = 1;
    }
}

void
InvSequencer::hide()
{
    shown_ = 0;
    XtVaSetValues(form,
                  XmNwidth, 1,
                  NULL);
    XtUnmanageChild(form);
}

//=========================================================================
//
// redefine those generic virtual functions
//
//=========================================================================
extern char *instance;
extern char *m_name;

const char *InvSequencer::getDefaultWidgetName() const
{
    return "InvSequencer";
}

const char *InvSequencer::getDefaultTitle() const
{
    static char buffer[512];
    sprintf(buffer, "%s_%s", m_name, instance);
    return buffer;
}

const char *InvSequencer::getDefaultIconTitle() const
{
    static char buffer[512];
    sprintf(buffer, "Seqencer_%s", instance);
    return buffer;
}

//======================================================================
//
// Description:
// define the pixmap icons
//
// Use: private
//
//======================================================================
void InvSequencer::definePixmaps(Widget parent)
{
    (void)parent;
    const char *pixname[] = { "REVERSE", "BACKWARD", "STOP", "FORWARD", "PLAY" };
    Pixel color;
    int i;

    // get foreground and background color
    XtVaGetValues(SoXt::getTopLevelWidget(), XmNforeground, &fg,
                  XmNbackground, &bg, NULL);

    // create bitmaps for recording
    for (i = 0; i < 5; i++)
    {
        std::string n("share/covise/bitmaps/");
        n += pixname[i];
        n += ".xbm";
        char *filename = coDirectory::canonical(n.c_str());
        //std::cerr << "looking for " << filename << std::endl;
        if (i == 2)
            color = fg;
        else
            color = fg;
        pix_records[i] = XmGetPixmap(XtScreen(SoXt::getTopLevelWidget()), filename, color, bg);
        if (pix_records[i] == XmUNSPECIFIED_PIXMAP)
            cerr << "Can't find bitmap: " << pixname[i] << std::endl;
    }
}

//======================================================================
//
// Description:
// called by the idle sensor when triggered
//
// Use: private
//
//======================================================================
void InvSequencer::thisAnimCB(SoSensor *)
{

    saveState();

    // re-schedule for animation
    animSensor->schedule();

    // let's see where the animation is going
    if (seqActive == INACTIVE)
    {
        // should not happen !
        animSensor->unschedule();
        return;
    };

    if (seqState == PLAYBACKWARD)
    {
        val--;
    }
    else if (seqState == PLAYFORWARD)
    {
        val++;
    }
    else
    {
        // should not happen
        animSensor->unschedule();
        return;
    }

    // cycle through
    if (val < boundmin)
        val = boundmax;
    if (val > boundmax)
        val = boundmin;

    // set widget
    XtVaSetValues(slider, XmNvalue, val, NULL);

    // call back the application
    if (valueChangedCallback != NULL)
        callValueChangedCallback(valueChangedUserData, valueChangedCallbackData);
}

//======================================================================
//
// Description:
// called by the idle sensor when triggered
//
// Use: private, static
//
//======================================================================
void InvSequencer::AnimCB(void *sequencer, SoSensor *idle)
{
    InvSequencer *seq = (InvSequencer *)sequencer;
    seq->thisAnimCB(idle);
}

//======================================================================
//
// Description:
// called by the list widget
//
// Use: private, static
//
//======================================================================
void
InvSequencer::inputCB(Widget w, int num, XmAnyCallbackStruct *list_data)
{
    (void)list_data;
    InvSequencer *sequencer;
    int value;
    char *charval;
    char Buffer[10];

    XtVaGetValues(w, XmNuserData, &sequencer, NULL);
    charval = XmTextGetString(w);
    if (charval == NULL)
        return;

    value = atoi(XmTextGetString(w));

    if (num == 10) // minimum was set
    {
        // out of range
        if (value < sequencer->min || value > sequencer->max)
        {
            sprintf(Buffer, "%d", sequencer->min);
            XtVaSetValues(sequencer->labelmin, XmNvalue, Buffer, NULL);
            XtVaSetValues(sequencer->slider, XmNminimum, sequencer->min, NULL);
            sequencer->boundmin = sequencer->min;
        }
        else
        {
            sequencer->boundmin = value;
            XtVaSetValues(sequencer->slider, XmNminimum, value, NULL);
            if (sequencer->val < sequencer->boundmin) // re-position slider
            {
                XtVaSetValues(sequencer->slider, XmNvalue, sequencer->boundmin, NULL);
                sequencer->val = sequencer->boundmin;
            }
        }
    }

    else if (num == 11) // maximum was set
    {
        // out of range
        if (value > sequencer->max || value < sequencer->min)
        {
            sprintf(Buffer, "%d", sequencer->max);
            XtVaSetValues(sequencer->labelmax, XmNvalue, Buffer, NULL);
            XtVaSetValues(sequencer->slider, XmNmaximum, sequencer->max, NULL);
            sequencer->boundmax = sequencer->max;
        }
        else
        {
            sequencer->boundmax = value;
            XtVaSetValues(sequencer->slider, XmNmaximum, value, NULL);
            if (sequencer->val > sequencer->boundmax) // re-position slider
            {
                XtVaSetValues(sequencer->slider, XmNvalue, sequencer->boundmax, NULL);
                sequencer->val = sequencer->boundmax;
            }
        }
    }

    // call back the application
    if (sequencer->valueChangedCallback != NULL)
        sequencer->callValueChangedCallback(sequencer->valueChangedUserData, sequencer->valueChangedCallbackData);
}

// stop explicit
void
InvSequencer::stop()
{
    saveState();
    seqState = STOP;
    seqActive = INACTIVE;
    animSensor->unschedule();
}

// start "PLAY" explicit
void
InvSequencer::play(int state)
{
    saveState();
    seqState = state;
    seqActive = ACTIVE;
    animSensor->schedule();
}

// Snap all Timesteps

void InvSequencer::snap()
{
    int old = val;
    val = min;
    char filename[64];
    char mask[16];

    if (max - min < 10000)
        strcpy(mask, "snap%04d.tiff");
    else if (max - min < 100000000)
        strcpy(mask, "snap%08d.tiff");
    else
        strcpy(mask, "snap%d.tiff");

    while (val <= max)
    {
        XtVaSetValues(slider, XmNvalue, val, NULL);
        callValueChangedCallback(valueChangedUserData, valueChangedCallbackData);
        sprintf(filename, mask, val - min);
        coviseViewer->snap(filename);
        val++;
    }
    val = old;

    // call back the application
    if (valueChangedCallback != NULL)
        callValueChangedCallback(valueChangedUserData, valueChangedCallbackData);

    XtVaSetValues(slider, XmNvalue, val, NULL);
}

//======================================================================
//
// Description:
// called by the list widget
//
// Use: private, static
//
//======================================================================
void InvSequencer::sequencerCB(Widget w, int num, XmPushButtonCallbackStruct *list_data)
{
    (void)list_data;
    InvSequencer *sequencer;

    XtVaGetValues(w, XmNuserData, &sequencer, NULL);

    sequencer->saveState();

    // STOP pressed  //
    if (num == 0)
    {
        sequencer->seqState = STOP;
        sequencer->seqActive = INACTIVE;
        sequencer->animSensor->unschedule();
        // call back the application
        if (sequencer->valueChangedCallback != NULL)
        {
            sequencer->callValueChangedCallback(sequencer->valueChangedUserData, sequencer->valueChangedCallbackData);
        }
        return;
    }

    // PLAYBACKWARD pressed
    if (num == -2)
    {
        // INACTIVE -> PLAYBACKWARD
        if (sequencer->seqActive == INACTIVE)
        {
            sequencer->seqState = PLAYBACKWARD;
            sequencer->seqActive = ACTIVE;
            sequencer->animSensor->schedule();
            (sequencer->val)--;
        }
        else // ignore
            return;
    }
    // BACKWARD pressed
    else if (num == -1)
    {
        sequencer->seqState = BACKWARD;
        sequencer->seqActive = INACTIVE;
        (sequencer->val)--;
    }
    // FORWARD pressed
    else if (num == 1)
    {
        sequencer->seqState = FORWARD;
        sequencer->seqActive = INACTIVE;
        (sequencer->val)++;
    }
    // PLAYFORWARD pressed
    else if (num == 2)
    {

        // INACTIVE -> PLAYFORWARD
        if (sequencer->seqActive == INACTIVE)
        {
            sequencer->seqState = PLAYFORWARD;
            sequencer->seqActive = ACTIVE;
            sequencer->animSensor->schedule();
            (sequencer->val)++;
        }
        else // ignore
            return;
    }

    if (sequencer->val < sequencer->boundmin)
        sequencer->val = sequencer->boundmin;
    if (sequencer->val > sequencer->boundmax)
        sequencer->val = sequencer->boundmax;

    // set new value //
    XtVaSetValues(sequencer->slider, XmNvalue, sequencer->val, NULL);

    // call back the application
    if (sequencer->valueChangedCallback != NULL)
        sequencer->callValueChangedCallback(sequencer->valueChangedUserData, sequencer->valueChangedCallbackData);
}

//======================================================================
//
// Description:
// set the sequencer inactive for user actions (slave renderers)
//
// Use: public
//
//======================================================================
void InvSequencer::setInactive()
{
    for (int i = 0; i < 5; i++)
        XtVaSetValues(label[i], XmNsensitive, False, NULL);
    XtVaSetValues(labelmin, XmNsensitive, False, NULL);
    XtVaSetValues(labelmax, XmNsensitive, False, NULL);
    XtVaSetValues(slider, XmNsensitive, False, NULL);
}

//======================================================================
//
// Description:
// set the sequencer active for user actions (master renderers)
//
// Use: public
//
//======================================================================
void InvSequencer::setActive()
{

    for (int i = 0; i < 5; i++)
        XtVaSetValues(label[i], XmNsensitive, True, NULL);
    XtVaSetValues(labelmin, XmNsensitive, True, NULL);
    XtVaSetValues(labelmax, XmNsensitive, True, NULL);
    XtVaSetValues(slider, XmNsensitive, True, NULL);
}

//======================================================================
//
// Description:
// set the sequencer active for user actions (master renderers)
//
// Use: public
//
//======================================================================
void InvSequencer::setSyncMode(char *message, int isMaster)
{

    if (strcmp("LOOSE", message) == 0)
    {
        sync_flag = SYNC_LOOSE;
        // update menu availability for loose coupling
        if (isMaster)
            setActive();
        else
            setActive();
    }
    else if (strcmp("SYNC", message) == 0)
    {
        sync_flag = SYNC_SYNC;
        if (isMaster)
            setActive();
        else
            setInactive();
    }
    else
    {
        sync_flag = SYNC_TIGHT;
        if (isMaster)
            setActive();
        else
            setInactive();
    }
}

//======================================================================
//
// Description:
// set the value changed callback
//
// Use: public
//
//======================================================================
void InvSequencer::setValueChangedCallback(SequencerCallback *func, void *userData)
{
    valueChangedCallback = func;
    valueChangedUserData = userData;
    valueChangedCallbackData = NULL; // not used
}

//======================================================================
//
// Description:
// remove the value changed callback
//
// Use: public
//
//======================================================================
void InvSequencer::removeValueChangedCallback()
{
    valueChangedCallback = NULL;
    valueChangedUserData = NULL;
    valueChangedCallbackData = NULL; // not used
}

//======================================================================
//
// Description:
// called by the list widget
//
// Use: private, static
//
//======================================================================

void InvSequencer::sliderCB(Widget w, void *userData, XmAnyCallbackStruct *data)
{
    (void)data;
    InvSequencer *seq = (InvSequencer *)userData;

    XtVaGetValues(w, XmNvalue, &seq->val, NULL);

    if (seq->valueChangedCallback != NULL)
        seq->callValueChangedCallback(seq->valueChangedUserData, seq->valueChangedCallbackData);
}

//======================================================================
//
// Description:
// called by the list widget
//
// Use: public
//
//======================================================================
void
InvSequencer::setSliderBounds(int minimum, int maximum, int value, int slidermin, int slidermax)
{

    char Buffer[10];
    // get new value //

    val = value;
    min = minimum;
    max = maximum;
    boundmin = slidermin;
    boundmax = slidermax;

    sprintf(Buffer, "%d", val);
    XtVaSetValues(slider, XmNvalue, val, NULL);

    sprintf(Buffer, "%d", slidermin);
    XtVaSetValues(labelmin, XmNvalue, Buffer, NULL);

    XtVaSetValues(slider, XmNminimum, min, NULL);

    sprintf(Buffer, "%d", boundmax);
    XtVaSetValues(labelmax, XmNvalue, Buffer, NULL);

    XtVaSetValues(slider, XmNmaximum, max, NULL);

    // call back the application
    //     if (valueChangedCallback != NULL) {
    // 	callValueChangedCallback(valueChangedUserData,valueChangedCallbackData);
    //     }
}

//********************** Test the shit ! **************************
/*
void main(int argc, char *argv[])
{

 Widget myWindow = SoXt::init(argv[0]);

 InvSequencer *seq = new InvSequencer(myWindow);
 seq->setSliderBounds(0,5,1);

 seq->show();

SoXt::show(myWindow);
SoXt::mainLoop();
}
*/
