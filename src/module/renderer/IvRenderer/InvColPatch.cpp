/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* $Log: InvColPatch.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#if DEBUG
#include <covise/covise.h>
#endif

#include "InvUIRegion.h"
#include "InvColorPatch.h"

#include <GL/gl.h>

/*
 * Defines
 */

#define SIDE (UI_THICK + 2 + UI_THICK)

////////////////////////////////////////////////////////////////////////
//
// Public constructor - build the widget right now
//
MyColorPatch::MyColorPatch(
    Widget parent,
    const char *name,
    SbBool buildInsideParent)
    : SoXtGLWidget(
          parent,
          name,
          buildInsideParent,
          SO_GLX_RGB,
          FALSE) // tell GLWidget not to build just yet
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component is what the app wants, so buildNow = TRUE
    constructorCommon(TRUE);
}

////////////////////////////////////////////////////////////////////////
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
MyColorPatch::MyColorPatch(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    SbBool buildNow)
    : SoXtGLWidget(
          parent,
          name,
          buildInsideParent,
          SO_GLX_RGB,
          FALSE) // tell GLWidget not to build just yet
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
    // as it was passed to us.
    constructorCommon(buildNow);
}

////////////////////////////////////////////////////////////////////////
//
// Called by the constructors
//
// private
//
void
MyColorPatch::constructorCommon(SbBool buildNow)
//
//////////////////////////////////////////////////////////////////////
{
    // init local vars
    color[0] = color[1] = color[2] = 0;
    setGlxSize(SbVec2s(40, 40)); // default size

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }
}

////////////////////////////////////////////////////////////////////////
//
//    Dummy virtual destructor.
//
MyColorPatch::~MyColorPatch()
//
////////////////////////////////////////////////////////////////////////
{
}

////////////////////////////////////////////////////////////////////////
//
//  Routine which draws the color patch
//
// Use: virtual public

void
MyColorPatch::redraw()
//
////////////////////////////////////////////////////////////////////////
{
    if (!isVisible())
        return;

    glXMakeCurrent(getDisplay(), getNormalWindow(), getNormalContext());

    // draw border
    SbVec2s size = getGlxSize();
    drawDownUIRegion(0, 0, size[0] - 1, size[1] - 1);

    // draw the patch color
    glColor3fv(color.getValue());
    glRecti(SIDE, SIDE, size[0] - SIDE, size[1] - SIDE);
}

////////////////////////////////////////////////////////////////////////
//
//  Sets the patch color
//
// Use: public

void
MyColorPatch::setColor(const SbColor &rgb)
//
////////////////////////////////////////////////////////////////////////
{
    // save color
    color = rgb;

    // now show the color change
    if (!isVisible())
        return;
    glXMakeCurrent(getDisplay(), getNormalWindow(), getNormalContext());

    glColor3fv(color.getValue());
    SbVec2s size = getGlxSize();
    glRecti(SIDE, SIDE, size[0] - SIDE, size[1] - SIDE);
}

////////////////////////////////////////////////////////////////////////
//
//	This routine is called when the window size has changed.
//
// Use: virtual private

void
MyColorPatch::sizeChanged(const SbVec2s &newSize)
//
////////////////////////////////////////////////////////////////////////
{
    // reset projection
    glXMakeCurrent(getDisplay(), getNormalWindow(), getNormalContext());
    glViewport(0, 0, newSize[0], newSize[1]);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, newSize[0], 0, newSize[1], -1, 1);
}
