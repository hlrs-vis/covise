/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_XT_COLOR_PATCH_
#define _INV_XT_COLOR_PATCH_

/* $Id: InvColorPatch.h,v 1.1 1994/04/12 13:39:31 zrfu0125 Exp zrfu0125 $ */

/* $Log: InvColorPatch.h,v $
 * Revision 1.1  1994/04/12  13:39:31  zrfu0125
 * Initial revision
 * */

#include <Inventor/Xt/SoXtGLWidget.h>
#include <Inventor/SbColor.h>

//////////////////////////////////////////////////////////////////////////////
//
//  Class: MyColorPatch
//
//	This class simply draws a 3D looking patch of color.
//
//////////////////////////////////////////////////////////////////////////////

// C-api: prefix=SoXtColPatch
class MyColorPatch : public SoXtGLWidget
{

public:
    MyColorPatch(
        Widget parent = NULL,
        const char *name = NULL,
        SbBool buildInsideParent = TRUE);
    ~MyColorPatch();

    //
    // set/get routines to specify the patch top color
    //
    // C-api: name=setCol
    void setColor(const SbColor &rgb);
    // C-api: name=getCol
    const SbColor &getColor()
    {
        return color;
    }

protected:
    // This constructor takes a boolean whether to build the widget now.
    // Subclasses can pass FALSE, then call buildWidget()
    // when they are ready for it to be built.
    SoEXTENDER
    MyColorPatch(
        Widget parent,
        const char *name,
        SbBool buildInsideParent,
        SbBool buildNow);

private:
    // redefine to do ColorPatch specific things
    virtual void redraw();
    virtual void sizeChanged(const SbVec2s &newSize);

    // local variables
    SbColor color;

    // this is called by both constructors
    void constructorCommon(SbBool buildNow);
};
#endif // _INV_XT_COLOR_PATCH_
