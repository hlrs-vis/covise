/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


/* $Log: InvColSlider.C,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#ifdef DEBUG
#include <stdio.h>
#endif

#include <X11/StringDefs.h>
#include <X11/Intrinsic.h>
#include <Xm/Text.h>

#include <Inventor/SbLinear.h>
#include <Inventor/SbColor.h>
#include "InvUIRegion.h"
#include "InvColorSlider.h"

#include <GL/gl.h>

////////////////////////////////////////////////////////////////////////
//
// Public constructor - build the widget right now
//
MyColorSlider::MyColorSlider(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    MyColorSlider::Type type)
    : MySlider(
          parent,
          name,
          buildInsideParent,
          FALSE) // tell slider not to build just yet
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component is what the app wants, so buildNow = TRUE
    constructorCommon(type, TRUE);
}

////////////////////////////////////////////////////////////////////////
//
// SoEXTENDER constructor - the subclass tells us whether to build or not
//
MyColorSlider::MyColorSlider(
    Widget parent,
    const char *name,
    SbBool buildInsideParent,
    MyColorSlider::Type type,
    SbBool buildNow)
    : MySlider(
          parent,
          name,
          buildInsideParent,
          FALSE) // tell slider not to build just yet
//
////////////////////////////////////////////////////////////////////////
{
    // In this case, this component may be what the app wants,
    // or it may want a subclass of this component. Pass along buildNow
    // as it was passed to us.
    constructorCommon(type, buildNow);
}

////////////////////////////////////////////////////////////////////////
//
// Called by the constructors
//
// private
//
void
MyColorSlider::constructorCommon(MyColorSlider::Type t, SbBool buildNow)
//
//////////////////////////////////////////////////////////////////////
{
    // init local vars
    WYSIWYGmode = FALSE;
    type = t;
    defaultColors = colors = NULL;
    geometry = NULL;
    color[0] = color[1] = color[2] = 0.0;

    // allocate memory for the color shading and geometry
    switch (type)
    {
    case RED_SLIDER:
    case GREEN_SLIDER:
    case BLUE_SLIDER:
        colors = new SbColor[2];
        defaultColors = new SbColor[2];
        geometry = new SbVec2f[4];
        break;
    case HUE_SLIDER:
        colors = new SbColor[7];
        defaultColors = new SbColor[7];
        geometry = new SbVec2f[14];
        break;
    case SATURATION_SLIDER:
    case VALUE_SLIDER:
    case INTENSITY_SLIDER:
        colors = new SbColor[2];
        defaultColors = colors; // since there is no fix default color
        geometry = new SbVec2f[4];
        break;
    }

    // ??? reset the geometry to prevent the window manager under PI a362
    // ??? to be killed (redrawing occurs even when window hasn't been
    // ??? mapped yet so those values would be bogus)
    int num = (type == HUE_SLIDER) ? 7 : 2;
    for (int i = 0; i < num; i++)
    {
        geometry[2 * i].setValue(0, 0);
        geometry[2 * i + 1].setValue(0, 0);
    }

    // now create the default colors
    makeDefaultColors();

    // add a callback to update the slider's baseColor whenever the
    // thumb is moved (unless it is INTENSITY_SLIDER, in which case
    // the base color is independent).
    if (type != INTENSITY_SLIDER)
        addValueChangedCallback(MyColorSlider::sliderChangedCB, this);

    // Build the widget tree, and let SoXtComponent know about our base widget.
    if (buildNow)
    {
        Widget w = buildWidget(getParentWidget());
        setBaseWidget(w);
    }
}

////////////////////////////////////////////////////////////////////////
//
//    Destructor.
//

MyColorSlider::~MyColorSlider()
//
////////////////////////////////////////////////////////////////////////
{
    // check to make sure colors and defaultColors are not the same
    if (defaultColors != colors)
        delete[] defaultColors;
    delete[] colors;
    delete[] geometry;
}

////////////////////////////////////////////////////////////////////////
//
//    This routine is used to specify the slider top color, which is
//  reflected differently depending on the slider type.
//
// usage: public

void
MyColorSlider::setBaseColor(
    const float c[3])
//
////////////////////////////////////////////////////////////////////////
{
    SbBool topChanged = true; // TRUE if the slider top region colors changed
    float newVal = getValue();

    switch (type)
    {

    case RED_SLIDER:
    case HUE_SLIDER:
        topChanged = (WYSIWYGmode && (c[1] != color[1] || c[2] != color[2]));
        newVal = c[0];
        break;

    case GREEN_SLIDER:
        topChanged = (WYSIWYGmode && (c[0] != color[0] || c[2] != color[2]));
        newVal = c[1];
        break;

    case BLUE_SLIDER:
        topChanged = (WYSIWYGmode && (c[0] != color[0] || c[1] != color[1]));
        newVal = c[2];
        break;

    case SATURATION_SLIDER:
        topChanged = (c[0] != color[0] || (WYSIWYGmode && c[2] != color[2]));
        newVal = c[1];
        break;

    case VALUE_SLIDER:
        topChanged = (c[0] != color[0] || (WYSIWYGmode && c[1] != color[1]));
        newVal = c[2];
        break;

    case INTENSITY_SLIDER:
        topChanged = (c[0] != color[0] || c[1] != color[1] || c[2] != color[2]);
        // thumb is independent of color for intensity sliders
        break;
    }

    // assign new color
    color[0] = c[0];
    color[1] = c[1];
    color[2] = c[2];

    if (topChanged)
        makeWYSIWYGcolors();

    // check for redraw needs
    if (newVal != getValue())
        setValue(newVal);
    else if (topChanged)
    {

        // only redraw the top region of the slider

        if (!isVisible())
            return;
        glXMakeCurrent(getDisplay(), getNormalWindow(), getNormalContext());
        drawSliderTopRegion();
    }
}

////////////////////////////////////////////////////////////////////////
//
//    This routine sets the WYSIWYG mode
//
// usage: public

void
MyColorSlider::setWYSIWYG(SbBool flag)
//
////////////////////////////////////////////////////////////////////////
{
    if (WYSIWYGmode == flag)
        return;

    WYSIWYGmode = flag;

    // build WYSIWYG colors
    if (WYSIWYGmode || type == SATURATION_SLIDER || type == VALUE_SLIDER)
        makeWYSIWYGcolors();

    // only need to redraw top region
    if (!isVisible())
        return;
    glXMakeCurrent(getDisplay(), getNormalWindow(), getNormalContext());
    drawSliderTopRegion();
}

////////////////////////////////////////////////////////////////////////
//
//    This routine is called when window has changed size
//
// usage: virtual protected

void
MyColorSlider::sizeChanged(const SbVec2s &newSize)
//
////////////////////////////////////////////////////////////////////////
{
    // first call base class routine
    MySlider::sizeChanged(newSize);

    SbVec2f *v = geometry;

    // now recompute things that depend on window size
    switch (type)
    {
    case HUE_SLIDER:
    {
        // assign y values
        int i;
        for (i = 0; i < 7; i++)
        {
            (*(v++))[1] = sly1;
            (*(v++))[1] = sly2 + 1;
        }

        // assign x values
        float l = (slx2 - slx1 + 1) / 6.0;
        v = geometry;
        (*(v++))[0] = slx1;
        (*(v++))[0] = slx1;
        for (i = 1; i < 6; i++)
        {
            (*(v++))[0] = slx1 + i * l;
            (*(v++))[0] = slx1 + i * l;
        }
        (*(v++))[0] = slx2 + 1;
        (*v)[0] = slx2 + 1;
    }
    break;

    default: // everything else
        (*v)[0] = (*(v + 1))[0] = slx1;
        (*(v + 2))[0] = (*(v + 3))[0] = slx2 + 1;
        (*v)[1] = (*(v + 3))[1] = sly1;
        (*(v + 1))[1] = (*(v + 2))[1] = sly2 + 1;
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
//    This routine draws the slider top region. It is called by the
//  MySlider::redraw routine.
//
// usage: virtual protected

void
MyColorSlider::drawSliderTopRegion()
//
////////////////////////////////////////////////////////////////////////
{
    SbColor *c = (WYSIWYGmode) ? colors : defaultColors;
    SbVec2f *v = geometry;

    switch (type)
    {
    case HUE_SLIDER:
    {
        glBegin(GL_QUAD_STRIP);
        for (int i = 0; i < 7; i++)
        {
            glColor3fv((c++)->getValue());
            glVertex2fv((v++)->getValue());
            glVertex2fv((v++)->getValue());
        }
        glEnd();
    }
    break;

    default: // everything else!
        glBegin(GL_POLYGON);
        glColor3fv((c++)->getValue());
        glVertex2fv((v++)->getValue());
        glVertex2fv((v++)->getValue());
        glColor3fv(c->getValue());
        glVertex2fv((v++)->getValue());
        glVertex2fv((v)->getValue());
        glEnd();
        break;
    }

    drawThumbUIRegion(thumx1, thumy1, thumx2, thumy2);
}

////////////////////////////////////////////////////////////////////////
//
//    Build the default colors (non WYSIWYG colors)
//
// usage: private

void
MyColorSlider::makeDefaultColors()
//
////////////////////////////////////////////////////////////////////////
{
    SbColor *c = defaultColors;

    switch (type)
    {
    case RED_SLIDER:
        (*c)[0] = (*c)[1] = (*c)[2] = 0.0;
        c++;
        (*c)[0] = 1.0;
        (*c)[1] = (*c)[2] = 0.0;
        break;

    case GREEN_SLIDER:
        (*c)[0] = (*c)[1] = (*c)[2] = 0.0;
        c++;
        (*c)[0] = (*c)[2] = 0.0;
        (*c)[1] = 1.0;
        break;

    case BLUE_SLIDER:
        (*c)[0] = (*c)[1] = (*c)[2] = 0.0;
        c++;
        (*c)[0] = (*c)[1] = 0.0;
        (*c)[2] = 1.0;
        break;

    case HUE_SLIDER:
        (*c)[0] = 1.0;
        (*c)[1] = (*c)[2] = 0.0;
        c++;
        (*c)[0] = (*c)[1] = 1.0;
        (*c)[2] = 0.0;
        c++;
        (*c)[0] = 0.0;
        (*c)[1] = 1.0;
        (*c)[2] = 0.0;
        c++;
        (*c)[0] = 0.0;
        (*c)[1] = (*c)[2] = 1.0;
        c++;
        (*c)[0] = (*c)[1] = 0.0;
        (*c)[2] = 1.0;
        c++;
        (*c)[0] = 1.0;
        (*c)[1] = 0.0;
        (*c)[2] = 1.0;
        c++;
        (*c)[0] = 1.0;
        (*c)[1] = (*c)[2] = 0.0;
        break;

    case SATURATION_SLIDER:
    case VALUE_SLIDER:
    case INTENSITY_SLIDER:
        c->setValue(0., 0., 0.);
        c++; // black...
        c->setValue(0., 0., 0.); // to black (initially)
        break;
    }
}

////////////////////////////////////////////////////////////////////////
//
//    builds the WYSIWYG colors.
//
// usage: private

void
MyColorSlider::makeWYSIWYGcolors()
//
////////////////////////////////////////////////////////////////////////
{
    SbColor *c = colors;

    switch (type)
    {
    case RED_SLIDER:
        (*c)[0] = 0.0;
        (*(c + 1))[0] = 1.0;
        (*c)[1] = (*(c + 1))[1] = color[1];
        (*c)[2] = (*(c + 1))[2] = color[2];
        break;

    case GREEN_SLIDER:
        (*c)[0] = (*(c + 1))[0] = color[0];
        (*c)[1] = 0.0;
        (*(c + 1))[1] = 1.0;
        (*c)[2] = (*(c + 1))[2] = color[2];
        break;

    case BLUE_SLIDER:
        (*c)[0] = (*(c + 1))[0] = color[0];
        (*c)[1] = (*(c + 1))[1] = color[1];
        (*c)[2] = 0.0;
        (*(c + 1))[2] = 1.0;
        break;

    case HUE_SLIDER:
    {
        // convert to rgb
        SbColor rgb;
        rgb.setHSVValue(color);

        // get max and min values
        float max = color[2];
        float min = (rgb[0] < rgb[1]) ? ((rgb[0] < rgb[2]) ? rgb[0] : rgb[2]) : ((rgb[1] < rgb[2]) ? rgb[1] : rgb[2]);

        // now assign the 7 color ranges
        (*c)[0] = max;
        (*c)[1] = (*c)[2] = min;
        c++;
        (*c)[0] = (*c)[1] = max;
        (*c)[2] = min;
        c++;
        (*c)[0] = min;
        (*c)[1] = max;
        (*c)[2] = min;
        c++;
        (*c)[0] = min;
        (*c)[1] = (*c)[2] = max;
        c++;
        (*c)[0] = (*c)[1] = min;
        (*c)[2] = max;
        c++;
        (*c)[0] = max;
        (*c)[1] = min;
        (*c)[2] = max;
        c++;
        (*c)[0] = max;
        (*c)[1] = (*c)[2] = min;
    }
    break;

    case SATURATION_SLIDER:
    {
        float val = (WYSIWYGmode) ? color[2] : 1.0;

        // find starting color (max value, grey)
        (*c)[0] = (*c)[1] = (*c)[2] = val;
        c++;

        // find ending color
        float hsv[3];
        hsv[0] = color[0];
        hsv[1] = 1.0;
        hsv[2] = val;
        c->setHSVValue(hsv);
    }
    break;

    case VALUE_SLIDER:
        c++; // only change the 2nd color (the 1st is always black)

        // find ending color
        if (color[1] == 0.0)
        {
            (*c)[0] = (*c)[1] = (*c)[2] = 1.0;
        }
        else
        {
            float hsv[3];
            hsv[0] = color[0];
            hsv[1] = (WYSIWYGmode) ? color[1] : 1.0;
            hsv[2] = 1.0;
            c->setHSVValue(hsv);
        }
        break;

    case INTENSITY_SLIDER:
        c++; // only change the 2nd color (the 1st is always black)
        *c = color;
        break;
    }
}

//
////////////////////////////////////////////////////////////////////////
// static callbacks stubs
////////////////////////////////////////////////////////////////////////
//

//
// called whenever the slider value changes. Make sure the
// corresponding baseColor factor is also updated to reflect
// the new slider position.
//
void
MyColorSlider::sliderChangedCB(void *pt, float val)
{
    MyColorSlider *p = (MyColorSlider *)pt;

    switch (p->type)
    {
    case RED_SLIDER:
    case HUE_SLIDER:
        p->color[0] = val;
        break;
    case GREEN_SLIDER:
    case SATURATION_SLIDER:
        p->color[1] = val;
        break;
    case BLUE_SLIDER:
    case VALUE_SLIDER:
        p->color[2] = val;
        break;
    case INTENSITY_SLIDER:
        break; // nothing to do since thumnb is indep of baseColor
    }
}
