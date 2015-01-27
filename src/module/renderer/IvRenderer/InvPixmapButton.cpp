/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//**************************************************************************
//
// * Description    : Pixmap button class
//
// * Class(es)      : InvPixmapButton
//
// * inherited from : none
//
// * Author  : Dirk Rantzau
//
// * History : 29.03.94 V 1.0
//
//**************************************************************************

#include <X11/Intrinsic.h>
#include <Xm/Xm.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>

#include "InvPixmapButton.h"

//    MESA hack - Do it better

#ifdef __linux__

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glx.h>
// extern "C" {
// GLXPixmap glXCreateGLXPixmapMESA( Display *dpy, XVisualInfo *visinfo, Pixmap pixmap, Colormap cmap )
// {
//         return glXCreateGLXPixmap( dpy, visinfo, pixmap);
// }
// }
#endif

////////////////////////////////////////////////////////////////////////
//
//  Constructor
//
InvPixmapButton::InvPixmapButton(Widget p, SbBool canSelect)
//
////////////////////////////////////////////////////////////////////////
{
    parent = p;
    selectFlag = FALSE;
    selectable = canSelect;
    normalPixmap = armPixmap = selectPixmap = 0;

    // Create the push button
    Arg args[8];
    int n = 0;
    XtSetArg(args[n], XmNmarginHeight, 0);
    n++;
    XtSetArg(args[n], XmNmarginWidth, 0);
    n++;
    XtSetArg(args[n], XmNshadowThickness, 2);
    n++;
    XtSetArg(args[n], XmNhighlightThickness, 0);
    n++;
    widget = XmCreatePushButtonGadget(parent, (char *)"", args, n);
}

////////////////////////////////////////////////////////////////////////
//
//  Destructor
//
InvPixmapButton::~InvPixmapButton()
//
////////////////////////////////////////////////////////////////////////
{
    //??? destroy the widget?
}

////////////////////////////////////////////////////////////////////////
//
//  Highlight the pixmap button.
//
//  Usage: public
//
void
InvPixmapButton::select(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (selectFlag == flag || !selectable)
        return;

    selectFlag = flag;

    XtVaSetValues(widget, XmNlabelPixmap,
                  selectFlag ? selectPixmap : normalPixmap, NULL);
}

////////////////////////////////////////////////////////////////////////
//
//  This routine builds the pixmaps (label pixmap and arm pixmap).
//
//  Usage: public
//
void
InvPixmapButton::setIcon(char *icon, int width, int height)
//
////////////////////////////////////////////////////////////////////////
{
    Display *display = XtDisplay(parent);
    Drawable d = DefaultRootWindow(display);
    int depth = XDefaultDepthOfScreen(XtScreen(parent));
    Pixel fg, bg, hl, abg;
    Arg args[8];
    int n;

    // get the color of the push buttons
    // ??? the foregrounf and background color have to be
    // ??? taken from the parent widget because we are using
    // ??? Gadget push buttons (not Widget push buttons)
    n = 0;
    XtSetArg(args[n], XmNforeground, &fg);
    n++;
    XtSetArg(args[n], XmNbackground, &bg);
    n++;
    XtSetArg(args[n], XmNtopShadowColor, &hl);
    n++; //??? highlight color
    XtGetValues(parent, args, n);

    n = 0;
    XtSetArg(args[n], XmNarmColor, &abg);
    n++;
    XtGetValues(widget, args, n);

    // free the old pixmaps
    if (normalPixmap)
        XFreePixmap(display, normalPixmap);
    if (armPixmap)
        XFreePixmap(display, armPixmap);
    if (selectPixmap)
        XFreePixmap(display, selectPixmap);
    normalPixmap = armPixmap = selectPixmap = 0;

    // create the pixmaps from the bitmap data (depth 1).
    normalPixmap = XCreatePixmapFromBitmapData(display, d,
                                               icon, width, height, fg, bg, depth);
    armPixmap = XCreatePixmapFromBitmapData(display, d,
                                            icon, width, height, fg, abg, depth);
    if (selectable)
        selectPixmap = XCreatePixmapFromBitmapData(display, d,
                                                   icon, width, height, fg, hl, depth);

    // now assign the pixmaps
    n = 0;
    XtSetArg(args[n], XmNlabelType, XmPIXMAP);
    n++;
    XtSetArg(args[n], XmNlabelPixmap,
             selectFlag ? selectPixmap : normalPixmap);
    n++;
    XtSetArg(args[n], XmNarmPixmap, armPixmap);
    n++;
    XtSetValues(widget, args, n);
}
