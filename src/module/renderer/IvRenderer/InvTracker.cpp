/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* $Log: InvTracker.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

//**************************************************************************
//
// * Description    : Headtracking extension to Inventor
//
// * Class(es)      :
//
// * inherited from :  InvFullViewer
//
// * Author  : Dirk Rantzau, Uwe Woessner
//
// * History : 09.09.94 V 1.0
//
//**************************************************************************

#include <stdio.h>
#include <math.h>

#include <X11/StringDefs.h>
#include <X11/Intrinsic.h>
#include <X11/Shell.h>

#include <Xm/Xm.h>
#include <Xm/LabelG.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/SeparatoG.h>
#include <Xm/CascadeB.h>
#include <Xm/CascadeBG.h>
#include <Xm/Form.h>
#include <Xm/ToggleB.h>
#include <Xm/ToggleBG.h>
#include <Xm/RowColumn.h>
#include <Xm/Scale.h>
#include <Xm/Text.h>

#include "logidrvr.h"

#ifndef __sgi
#include "ThumbWheel.h"
#else
#include <Sgm/ThumbWheel.h>
#endif

#include <Xm/MessageB.h>

#include <Inventor/SoPickedPoint.h>
#include <Inventor/Xt/SoXt.h>
#include <Inventor/Xt/SoXtResource.h>
#include "InvExaminerViewer.h"
#include "InvFullViewer.h"
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <GL/gl.h>
#include "InvPixmapButton.h"
#include <Inventor/errors/SoDebugError.h>
#include <Inventor/sensors/SoTimerSensor.h>

#include <util/coLog.h>

#include "InvTracker.h"
#include "polhemusdrvr.h"

/* Uwe Woessner (Added TRACK_PUSH)*/
// list of custom push buttons
enum ViewerPushButtons
{
    PICK_PUSH,
    VIEW_PUSH,
    TRACK_PUSH,
    HELP_PUSH,
    HOME_PUSH,
    SET_HOME_PUSH,
    VIEW_ALL_PUSH,
    SEEK_PUSH,

    PUSH_NUM
};

////////////////////////////////////////////////////////////////////////
//
//	Called whenever the tracking sensor fires.
//
//  Use: static private
//
//  Uwe Woessner
//
void
InvFullViewer::trackingSensorCB(void *p, SoSensor *)
//
////////////////////////////////////////////////////////////////////////
{
    InvFullViewer *v = (InvFullViewer *)p;
    MouseRecordType record;
    PolhemusRecord p_record;
    float P, Y, R;
    float dP, dY, dR;
    float dpx, dpy, dpz;
    // static float angle = 0;
    static float tdpx = 0;
    static float tdpy = 0;
    static float tdpz = 0;
    static int out_of_range_flag = 0;
    // static float tdP=0;
    // static float tdY=0;
    // static float tdR=0;

    // SoCamera *camera = InvViewer::viewer->getCamera();

    // grab one valid record
    if (v->trackingDevice < 3) // Logitech on port 1 or 2
    {
        get_record(v->trackingFd, &record);
        // out of range
        if ((record.buttons & 0x20) && (!out_of_range_flag))
        {
            v->buttonList[TRACK_PUSH]->select(0);
            out_of_range_flag = 1;
        }
        else if (out_of_range_flag)
        {
            v->buttonList[TRACK_PUSH]->select(1);
            out_of_range_flag = 0;
        }

// move the camera position by the translation amount
// in the forward direction.
// we will ignore 'y' motion since this is the walk viewer.
// first, orient the tracker data to our camera orientation.

// first update the position according to current position and tracker position
#define TFACT 0.05

        // get current transform
        SoSFVec3f translation;
        // scale position and create relative position
        dpx = (record.x - tdpx) * TFACT;
        dpy = (record.y - tdpy) * TFACT;
        dpz = (record.z - tdpz) * TFACT;
        // printf("%f %f, %f\n",dpx,dpy,dpz);
        // set error boundary
        if ((fabs(dpx) > 10.0) || (fabs(dpy) > 10.0) || (fabs(dpz) > 10.0))
        {
            cerr << "You moved to fast!\n";
            // save values
            tdpx = record.x;
            tdpy = record.y;
            tdpz = record.z;
            return; // maybe the tracker track is not correct
        }
        // save values
        tdpx = record.x;
        tdpy = record.y;
        tdpz = record.z;

        // printf("r: %f  %f %f %f\n",dpx,dpy,dpz,record.xr,record.yr,record.zr,record.ar);
        translation.setValue(dpx, dpy, dpz);
        v->camera->position.setValue(v->camera->position.getValue() + translation.getValue());

        // now the rotations
        //

        // little correction
        P = (record.pitch + 20.0) * M_PI / 180.0;
        Y = (record.yaw) * M_PI / 180.0;
        R = (record.roll) * M_PI / 180.0;

#define RFACT 2.0
        // scale rotation (better not)
        dP = P * (RFACT);
        dY = Y * (RFACT);
        dR = R * (RFACT / 2.0);

        float ra = cos(dR / 2) * cos(dP / 2) * cos(dY / 2) + sin(dR / 2) * sin(dP / 2) * sin(dY / 2);
        float rx = cos(dR / 2) * sin(dP / 2) * cos(dY / 2) + sin(dR / 2) * cos(dP / 2) * sin(dY / 2);
        float ry = cos(dR / 2) * cos(dP / 2) * sin(dY / 2) + sin(dR / 2) * sin(dP / 2) * cos(dY / 2);
        float rz = sin(dR / 2) * cos(dP / 2) * cos(dY / 2) + cos(dR / 2) * sin(dP / 2) * sin(dY / 2);

        v->camera->orientation.setValue(rx, ry, rz, ra);
    }
    else // Fastrak is in the house (port 3 or 4)
    {
        fastrackGetSingleRecord(v->trackingFd, &p_record);
        SoSFVec3f translation;
        // scale position and create relative position
        dpx = (p_record.x - tdpx) * TFACT;
        dpy = (p_record.y - tdpy) * TFACT;
        dpz = (p_record.z - tdpz) * TFACT;
        // save values
        tdpx = p_record.x;
        tdpy = p_record.y;
        tdpz = p_record.z;

        // printf("r: %f  %f %f %f\n",p_record.q1,p_record.q2,p_record.q3,p_record.w);
        translation.setValue(dpx, dpy, dpz);
        v->camera->position.setValue(v->camera->position.getValue() + translation.getValue());

        // v->camera->orientation.setValue(p_record.q1,p_record.q2,p_record.q3,(float)(2*facos(p_record.w)));
    }
}

////////////////////////////////////////////////////////////////////////
//
//	Initializes the Tracking device and returns true if successfull.
//
//  Use: static private
//
//  Uwe Woessner
//
SbBool
InvFullViewer::initTracking(void)
//
////////////////////////////////////////////////////////////////////////
{
    // Open head tracker device

    /// commented out DRA 27.03.98 for release 4.5 (serial problems on 6.3,6.4->tracking support disabled)
    /*
      trackingFd=0;
      char buf[100];
      switch(trackingDevice)
      {
      case 1:
          print_comment( __LINE__,__FILE__,"Connecting to logitech device on port /dev/ttyd1...");
          if ((trackingFd = logitech_open ("/dev/ttyd1")) <= 0)
          {
              print_comment( __LINE__,__FILE__,"error opening serial port!");
         return(FALSE);
   }
   //  Set mouse mode
   print_comment( __LINE__,__FILE__,"Enabling mouse mode, euler records, demand reporting...");
   cu_headtracker_mode (trackingFd);
   cu_euler_mode (trackingFd);
   // use demand mode and timer sensor
   cu_demand_reporting (trackingFd);
   break;
   case 2:
   print_comment( __LINE__,__FILE__,"Connecting to logitech device on port /dev/ttyd2...");
   if ((trackingFd = logitech_open ("/dev/ttyd2")) <= 0)
   {
   print_comment( __LINE__,__FILE__,"error opening serial port!");
   return(FALSE);
   }
   //  Set mouse mode
   print_comment( __LINE__,__FILE__,"Enabling mouse mode, euler records, demand reporting...");
   cu_headtracker_mode (trackingFd);
   cu_euler_mode (trackingFd);
   // use demand mode and timer sensor
   cu_demand_reporting (trackingFd);
   break;
   case 3:
   case 4:
   case 5:
   case 6:
   sprintf(buf,"/dev/ttyd%d",trackingDevice-2);
   print_comment( __LINE__,__FILE__,"Connecting to fastrak device on port /dev/ttyd?...");
   if ((trackingFd = fastrackOpen (buf)) <= 0)
   {
   print_comment( __LINE__,__FILE__,"error opening serial port!");
   return(FALSE);
   }
   //  Set mouse mode
   // configuration
   fastrackReset(trackingFd);
   fastrackSetPositionFilter(trackingFd,0.05,0.2,0.8,0.8);
   fastrackSetAttitudeFilter(trackingFd,0.05,0.2,0.8,0.8);
   fastrackSetAsciiFormat(trackingFd);
   fastrackDisableContinuousOutput(trackingFd);
   // fastrackSetUnitToCentimeters(trackingFd);
   fastrackSetReferenceFrame(trackingFd,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0);
   fastrackSetHemisphere(trackingFd,0,0,-1.0);
   fastrackSetOutputToQuaternions(trackingFd);
   break;
   default:
   print_comment( __LINE__,__FILE__,"Device not supported");
   break;
   }
   print_comment( __LINE__,__FILE__,"...done");
   */
    return (TRUE);
}

////////////////////////////////////////////////////////////////////////
//
//	close tracking device
//
//  Use: static private
//
//  Uwe Woessner
//
void
InvFullViewer::closeTracking(void)
//
////////////////////////////////////////////////////////////////////////
{
    // close head tracker device
    close(trackingFd);
    covise::print_comment(__LINE__, __FILE__, "Releasing head tracking device");
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//	This creates the tracking rate preference sheet stuff.
//
// Use: protected
Widget
InvFullViewer::createTrackingPrefSheetGuts(Widget parent)
//
////////////////////////////////////////////////////////////////////////
{
    Widget label, thumb, text, device, devicem, devices[6];
    Arg args[12];
    XmString string;
    int n;

    // create a form to hold everything together
    Widget form = XtCreateWidget("", xmFormWidgetClass,
                                 parent, NULL, 0);

    // create device choice
    devicem = XmCreatePulldownMenu(form, (char *)"trackingmenu", NULL, 0);
    devices[0] = XmCreatePushButtonGadget(devicem, (char *)"Logitec on com1", NULL, 0);
    devices[1] = XmCreatePushButtonGadget(devicem, (char *)"Logitec on com2", NULL, 0);
    devices[2] = XmCreatePushButtonGadget(devicem, (char *)"Fastrak on com1", NULL, 0);
    devices[3] = XmCreatePushButtonGadget(devicem, (char *)"Fastrak on com2", NULL, 0);
    devices[4] = XmCreatePushButtonGadget(devicem, (char *)"Fastrak on com3", NULL, 0);
    devices[5] = XmCreatePushButtonGadget(devicem, (char *)"Fastrak on com4", NULL, 0);
    XtAddCallback(devices[0], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::trackingDevice1CB, (XtPointer) this);
    XtAddCallback(devices[1], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::trackingDevice2CB, (XtPointer) this);
    XtAddCallback(devices[2], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::trackingDevice3CB, (XtPointer) this);
    XtAddCallback(devices[3], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::trackingDevice4CB, (XtPointer) this);
    XtAddCallback(devices[4], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::trackingDevice5CB, (XtPointer) this);
    XtAddCallback(devices[5], XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::trackingDevice6CB, (XtPointer) this);
    XtManageChildren(devices, 6);
    string = XmStringCreate((char *)"xxTrackingdevices:", (XmStringCharSet)XmSTRING_DEFAULT_CHARSET);
    n = 0;
    XtSetArg(args[n], XmNlabelString, string);
    n++;
    XtSetArg(args[n], XmNsubMenuId, devicem);
    n++;
    device = XmCreateOptionMenu(form, (char *)"trackingoption", args, n);

    // create the label
    label = XtCreateWidget("Tracking rate:", xmLabelGadgetClass,
                           form, NULL, 0);

    // allocate the thumbwheel
    n = 0;
    XtSetArg(args[n], XmNvalue, 110);
    n++;
    XtSetArg(args[n], SgNangleRange, 720);
    n++;
    XtSetArg(args[n], XmNmaximum, 1010);
    n++;
    XtSetArg(args[n], XmNminimum, 10);
    n++;
    // XtSetArg(args[n], SgNunitsPerRotation, 1000); n++;
    XtSetArg(args[n], SgNshowHomeButton, FALSE);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    XtSetArg(args[n], XmNorientation, XmHORIZONTAL);
    n++;
    trackingWheel = thumb = SgCreateThumbWheel(form, (char *)"", args, n);

    XtAddCallback(thumb, XmNvalueChangedCallback,
                  (XtCallbackProc)InvFullViewer::trackingWheelCB, (XtPointer) this);
    // XtAddCallback(thumb, XmNdragCallback,
    //     (XtCallbackProc) InvFullViewer::trackingWheelCB, (XtPointer) this);

    // allocate the text field
    n = 0;
    char str[15];
    sprintf(str, "%.6f", trackingWheelVal);
    XtSetArg(args[n], XmNvalue, str);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 1);
    n++;
    XtSetArg(args[n], XmNcolumns, 8);
    n++;
    trackingField = text = XtCreateWidget("", xmTextWidgetClass,
                                          form, args, n);
    XtAddCallback(text, XmNactivateCallback,
                  (XtCallbackProc)InvFullViewer::trackingFieldCB,
                  (XtPointer) this);

    // layout

    n = 0;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_FORM);
    n++;
    XtSetValues(text, args, n);

    n = 0;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, text);
    n++;
    XtSetArg(args[n], XmNbottomOffset, 3);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNrightWidget, text);
    n++;
    XtSetArg(args[n], XmNrightOffset, 3);
    n++;
    XtSetValues(thumb, args, n);

    n = 0;
    XtSetArg(args[n], XmNbottomAttachment, XmATTACH_OPPOSITE_WIDGET);
    n++;
    XtSetArg(args[n], XmNbottomWidget, thumb);
    n++;
    XtSetArg(args[n], XmNrightAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNrightWidget, thumb);
    n++;
    XtSetArg(args[n], XmNrightOffset, 5);
    n++;
    XtSetValues(label, args, n);

    n = 0;
    XtSetArg(args[n], XmNleftAttachment, XmATTACH_FORM);
    n++;
    XtSetArg(args[n], XmNtopAttachment, XmATTACH_WIDGET);
    n++;
    XtSetArg(args[n], XmNtopWidget, thumb);
    n++;
    XtSetArg(args[n], XmNtopOffset, 3);
    n++;
    XtSetValues(device, args, n);

    // manage children
    XtManageChild(text);
    XtManageChild(thumb);
    XtManageChild(label);
    XtManageChild(device);
    return form;
}

void
InvFullViewer::trackingWheelCB(Widget, InvFullViewer *v, XtPointer *d)
{
    SgThumbWheelCallbackStruct *data = (SgThumbWheelCallbackStruct *)d;

    // reason = XmCR_VALUE_CHANGED
    v->trackingWheelVal = data->value / 1000.0;
    if (v->trackingFlag)
    {
        v->trackingSensor->unschedule();
        v->trackingSensor->setInterval((double)v->trackingWheelVal);
        v->trackingSensor->schedule();
    }

    // update the text field
    char str[15];
    sprintf(str, "%.6f", v->trackingWheelVal);
    XmTextSetString(v->trackingField, str);
}

void
InvFullViewer::trackingDevice1CB(Widget, InvFullViewer *v, XtPointer)
{

    v->trackingDevice = 1;
    if (v->trackingFlag)
    {
        v->closeTracking();
        v->initTracking();
    }
}

void
InvFullViewer::trackingDevice2CB(Widget, InvFullViewer *v, XtPointer)
{

    v->trackingDevice = 2;
    if (v->trackingFlag)
    {
        v->closeTracking();
        v->initTracking();
    }
}

void
InvFullViewer::trackingDevice3CB(Widget, InvFullViewer *v, XtPointer)
{

    v->trackingDevice = 3;
    if (v->trackingFlag)
    {
        v->closeTracking();
        v->initTracking();
    }
}

void
InvFullViewer::trackingDevice4CB(Widget, InvFullViewer *v, XtPointer)
{

    v->trackingDevice = 4;
    if (v->trackingFlag)
    {
        v->closeTracking();
        v->initTracking();
    }
}

void
InvFullViewer::trackingDevice5CB(Widget, InvFullViewer *v, XtPointer)
{

    v->trackingDevice = 5;
    if (v->trackingFlag)
    {
        v->closeTracking();
        v->initTracking();
    }
}

void
InvFullViewer::trackingDevice6CB(Widget, InvFullViewer *v, XtPointer)
{

    v->trackingDevice = 6;
    if (v->trackingFlag)
    {
        v->closeTracking();
        v->initTracking();
    }
}

void
InvFullViewer::trackingFieldCB(Widget field, InvFullViewer *v, void *)
{
    Arg args[12];
    int n;
    // get text value from the label and update camera
    char *str = XmTextGetString(field);
    float val;
    if (sscanf(str, "%f", &val) && val > 0)
    {
        v->trackingWheelVal = val;
        v->redraw();
    }
    XtFree(str);
    // Set new tracking rate
    if (v->trackingFlag)
    {
        v->trackingSensor->unschedule();
        v->trackingSensor->setInterval((double)v->trackingWheelVal);
        v->trackingSensor->schedule();
    }

    // update the thumbwheel
    n = 0;
    XtSetArg(args[n], XmNvalue, int(v->trackingWheelVal * 1000));
    n++;
    XtSetValues(v->trackingWheel, args, n);

    // reformat text field
    char valStr[10];
    sprintf(valStr, "%.6f", v->trackingWheelVal);
    XmTextSetString(field, valStr);

    // make the text field loose the focus
    XmProcessTraversal(SoXt::getShellWidget(field), XmTRAVERSE_CURRENT);
}
