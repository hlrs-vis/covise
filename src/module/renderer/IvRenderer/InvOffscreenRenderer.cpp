/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++               Implementation of class InvOffscreenRenderer          ++
// ++               enhanced and corrected version of SoOffscreenRenderer ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 21.10.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

// original  SGI open source Inventor comments:
// ================================= SGI's original header =============================================
/*
 *
 *  Copyright (C) 2000 Silicon Graphics, Inc.  All Rights Reserved.
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  Further, this software is distributed without any warranty that it is
 *  free of the rightful claim of any third person regarding infringement
 *  or the like.  Any license provided herein, whether implied or
 *  otherwise, applies only to this software file.  Patent licenses, if
 *  any, provided herein do not apply to combinations of this program with
 *  other software, or any other product whatsoever.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *  Contact information: Silicon Graphics, Inc., 1600 Amphitheatre Pkwy,
 *  Mountain View, CA  94043, or:
 *
 *  http://www.sgi.com
 *
 *  For further information regarding this notice, see:
 *
 *  http://oss.sgi.com/projects/GenInfo/NoticeExplan/
 *
 */

/*
 * Copyright (C) 1990,91,92,93   Silicon Graphics, Inc.
 *
 _______________________________________________________________________
 ______________  S I L I C O N   G R A P H I C S   I N C .  ____________
 |
 |   $Revision: 1.3 $
 |
 |   Classes:
 |	SoOffscreenRenderer
 |
|   Author(s)		: Dave Immel
|
______________  S I L I C O N   G R A P H I C S   I N C .  ____________
_______________________________________________________________________
*/
// ================================= END SGI's original header =========================================

#include <covise/covise.h>
#include <tiffio.h>

#define _HAS_UINT

#define DEBUG

#include <GL/gl.h>

#include "InvOffscreenRenderer.h"

// gross stuff to read images, grabbed from image.h and the image
// library.  Had to grab it since the real .h file is not c++ compatible.
extern "C" {

typedef struct
{
    unsigned short imagic; /* stuff saved on disk . . */
    unsigned short type;
    unsigned short dim;
    unsigned short xsize;
    unsigned short ysize;
    unsigned short zsize;
    uint32_t min;
    uint32_t max;
    uint32_t wastebytes;
    char name[80];
    uint32_t colormap;
} IMAGE;
extern IMAGE *fiopen(int, const char *, int = 0, int = 0,
                     int = 0, int = 0, int = 0);
extern void putrow(IMAGE *, short *, unsigned int, unsigned int);
extern void iclose(IMAGE *);
}; /* end of extern "C" */

#define TYPEMASK 0xff00
#define BPPMASK 0x00ff
#define ITYPE_VERBATIM 0x0000
#define ITYPE_RLE 0x0100
#define ISRLE(type) (((type)&0xff00) == ITYPE_RLE)
#define ISVERBATIM(type) (((type)&0xff00) == ITYPE_VERBATIM)
#define BPP(type) ((type)&BPPMASK)
#define RLE(bpp) (ITYPE_RLE | (bpp))
#define VERBATIM(bpp) (ITYPE_VERBATIM | (bpp))
#define IBUFSIZE(pixels) ((pixels + (pixels >> 6)) * sizeof(int32_t))
#define RLE_NOP 0x00

// Offscreen renderings will always be rendered as RGB images
// Currently the OpenGL does not support storing alpha channel information
// in the offscreen pixmap.
static int attributeList[] = {
    GLX_RGBA, GLX_RED_SIZE, 1, GLX_GREEN_SIZE, 1,
    GLX_BLUE_SIZE, 1, GLX_DEPTH_SIZE, 1, None
};

//
// KLUDGE!!!!!
// In order to keep caches from being blown away when the contexts
// is deleted, assign a unique number to them with the rendering action.
// Start at 10000 and increment.
//
static uint32_t SoContextIncrement = 10000;

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Constructor.
//
// Use: public

InvOffscreenRenderer::InvOffscreenRenderer(const SbViewportRegion &viewportRegion)

//
////////////////////////////////////////////////////////////////////////
{
    offAction = new SoGLRenderAction(viewportRegion);
    userAction = NULL;
    pixelBuffer = NULL;
    display = NULL;
    comps = InvOffscreenRenderer::RGB;
    backgroundColor.setValue(1.0, 1.0, 1.0);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Constructor.
//
// Use: public

InvOffscreenRenderer::InvOffscreenRenderer(
    SoGLRenderAction *act)

//
////////////////////////////////////////////////////////////////////////
{
    offAction = new SoGLRenderAction(act->getViewportRegion());
    userAction = act;
    pixelBuffer = NULL;
    display = NULL;
    comps = InvOffscreenRenderer::RGB;
    backgroundColor.setValue(1.0, 1.0, 1.0);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Destructor.
//
// Use: public

InvOffscreenRenderer::~InvOffscreenRenderer()

//
////////////////////////////////////////////////////////////////////////
{
    delete offAction;

    if (pixelBuffer != NULL)
        delete pixelBuffer;

    // Delete the pixmap, window, and context, as it is no longer needed
    if (display != NULL)
    {
        glXDestroyGLXPixmap(display, pixmap);
        glXDestroyContext(display, context);
        XCloseDisplay(display);
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Calculate and return the number of pixels per inch on the current
//    device horizontally.
//
// Use: static, public

float
InvOffscreenRenderer::getScreenPixelsPerInch()

//
////////////////////////////////////////////////////////////////////////
{
    Display *tmpDisplay;
    float pix;

    // Create an X Display
    if ((tmpDisplay = XOpenDisplay(NULL)) == NULL)
        return 75.0;

    // Get the dimensions of the screen
    pix = DisplayWidth(tmpDisplay, 0) * 25.4 / (float)DisplayWidthMM(tmpDisplay, 0);
    XCloseDisplay(tmpDisplay);

    return pix;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Return the maximum allowable resolution of a viewport in which
//    to render a scene.
//
// Use: static, public

SbVec2s
InvOffscreenRenderer::getMaximumResolution()

//
////////////////////////////////////////////////////////////////////////
{
    Display *dpy = NULL;
    XVisualInfo *vi;
    GLXContext cx;
    GLXPixmap glxpmap;
    Pixmap xpmap;
    SbVec2s tmpvp(2, 2);
    GLint params[2];

    if (!initPixmap(dpy, vi, cx, tmpvp, glxpmap, xpmap))
    {
#ifdef DEBUG
        SoDebugError::post("InvOffscreenRenderer::getMaximumResolution",
                           "Could not initialize Pixmap");
#endif
        return SbVec2s(-1, -1);
    }

    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, params);
    glXDestroyGLXPixmap(dpy, glxpmap);
    glXDestroyContext(dpy, cx);
    XCloseDisplay(dpy);

    return (SbVec2s((short)params[0], (short)params[1]));
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Set the viewport region.
//
// Use: public

void
InvOffscreenRenderer::setViewportRegion(
    const SbViewportRegion &viewportRegion)

//
////////////////////////////////////////////////////////////////////////
{
    offAction->setViewportRegion(viewportRegion);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Get the viewport region.
//
// Use: public

const SbViewportRegion &
InvOffscreenRenderer::getViewportRegion() const

//
////////////////////////////////////////////////////////////////////////
{
    return offAction->getViewportRegion();
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Set the render action.
//
// Use: public

void
InvOffscreenRenderer::setGLRenderAction(
    SoGLRenderAction *act)

//
////////////////////////////////////////////////////////////////////////
{
    userAction = act;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Get the render action.
//
// Use: public

SoGLRenderAction *
InvOffscreenRenderer::getGLRenderAction() const

//
////////////////////////////////////////////////////////////////////////
{
    return userAction ? userAction : offAction;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Render the given scene graph into a buffer.
//
// Use: public

SbBool
InvOffscreenRenderer::render(SoNode *scene)
//
////////////////////////////////////////////////////////////////////////
{
    // Delete the pixel buffer if it has been previously used.
    if (pixelBuffer != NULL)
    {
        delete pixelBuffer;
        pixelBuffer = NULL;
    }

    // Set the render action to use.
    SoGLRenderAction *act;
    if (userAction != NULL)
        act = userAction;
    else
        act = offAction;
    renderedViewport = act->getViewportRegion();

    // Get the offscreen pixmap ready for rendering
    if (!setupPixmap())
    {
#ifdef DEBUG
        SoDebugError::post("InvOffscreenRenderer::render",
                           "Could not setup Pixmap");
#endif

        return FALSE;
    }

    // Set the GL cache context for the action to a unique number
    // (we'll use and increment SoContextIncrement), so that it doesn't try
    // to use display lists from other contexts.
    uint32_t oldContext = act->getCacheContext();
    act->setCacheContext(SoContextIncrement++);
    act->apply(scene);
    act->setCacheContext(oldContext);

    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Render the given scene graph into a buffer.
//
// Use: public

SbBool
InvOffscreenRenderer::render(SoPath *scene)

//
////////////////////////////////////////////////////////////////////////
{
    // Delete the pixel buffer if it has been previously used.
    if (pixelBuffer != NULL)
    {
        delete pixelBuffer;
        pixelBuffer = NULL;
    }

    // Set the render action to use.
    SoGLRenderAction *act;
    if (userAction != NULL)
        act = userAction;
    else
        act = offAction;
    renderedViewport = act->getViewportRegion();

    // Get the offscreen pixmap ready for rendering
    if (!setupPixmap())
    {
#ifdef DEBUG
        SoDebugError::post("InvOffscreenRenderer::writeToRGB",
                           "Could not setup Pixmap");
#endif
        return FALSE;
    }

    // Render the scene into the new graphics context

    // Set the GL cache context for the action to a unique number
    // (we'll use and increment SoContextIncrement), so that it doesn't try
    // to use display lists from other contexts.
    uint32_t oldContext = act->getCacheContext();
    act->setCacheContext(SoContextIncrement++);
    act->apply(scene);
    act->setCacheContext(oldContext);

    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Return the rendered buffer.
//
// Use: public

unsigned char *
InvOffscreenRenderer::getBuffer() const

//
////////////////////////////////////////////////////////////////////////
{
    // If the buffer has not been gotten yet, read the pixels into
    // the buffer.  Return the buffer to the user.
    if (pixelBuffer == NULL)
    {

        if (!setContext())
            return NULL;
        ((InvOffscreenRenderer *)this)->readPixels();
    }

    return pixelBuffer;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Write the buffer into a .rgb file.
//
// Use: public

SbBool
InvOffscreenRenderer::writeToRGB(const char *filename) const

//
////////////////////////////////////////////////////////////////////////
{
    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        cerr << " Could not open file " << filename << endl;
        cerr << "InvOffscreenRenderer::writeToRGB" << endl;
        return FALSE;
    }

    // Set the graphics context to be the offscreen pixmap
    if (!setContext())
        return FALSE;

    // Open an image file for writing
    int dimensions, components;
    if (comps == (InvOffscreenRenderer::LUMINANCE_TRANSPARENCY) || (comps == InvOffscreenRenderer::RGB_TRANSPARENCY))
    {
#ifdef DEBUG
        SoDebugError::post("InvOffscreenRenderer::writeToRGB",
                           "The RGB file format does not support transparency");
#endif
        return FALSE;
    }
    else if (comps == InvOffscreenRenderer::LUMINANCE)
    {
        dimensions = 2;
        components = 1;
    }
    else
    {
        dimensions = 3;
        components = 3;
    }

    // Get the correct viewport size.
    const SbVec2s &vpSize = renderedViewport.getViewportSizePixels();

    int ifile = fileno(fp);
    IMAGE *image;

    if ((image = fiopen(ifile, "w", RLE(1), dimensions,
                        (unsigned int)vpSize[0], (unsigned int)vpSize[1],
                        components)) == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("InvOffscreenRenderer::writeToRGB",
                           "could not open image file");
#endif
        return FALSE;
    }

    // Get the format of the pixmap data
    GLenum format;
    getFormat(format);

    // For each row in the pixel buffer, write the row into the image file
    short *rowBuf = new short[vpSize[0]];
    unsigned char *pBuf = new unsigned char[vpSize[0] * components * 2];

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for (int row = 0; row < vpSize[1]; row++)
    {

        // Read the next scanline of pixels from the offscreen pixmap
        glReadPixels(0, row, vpSize[0], 1, format,
                     GL_UNSIGNED_BYTE, (GLvoid *)pBuf);

        // The pixel in the pixel buffer store pixel information arranged
        // by pixel, whereas the .rgb file stores pixel information arranged
        // by color component.  So scanlines of component data must be
        // accumulated before a row can be written.
        unsigned char *tbuf = pBuf;

        // Convert each color component
        for (int comp = 0; comp < components; comp++)
        {
            short *trow = rowBuf;

            // Convert a row
            tbuf = pBuf + comp;
            for (int j = 0; j < vpSize[0]; j++, tbuf += components)
                *trow++ = (short)*tbuf;
            putrow(image, rowBuf, row, comp);
        }
    }
    iclose(image);
    delete[] pBuf;
    delete[] rowBuf;
    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Write a hex value into Encapsulated PostScript.
//
// Use: private

void
InvOffscreenRenderer::putHex(
    FILE *fp,
    char val,
    int &hexPos)

//
////////////////////////////////////////////////////////////////////////
{
    fprintf(fp, "%02hx", (unsigned short)(unsigned char)val);
    if (++hexPos >= 32)
    {
        fprintf(fp, "\n");
        hexPos = 0;
    }
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Write the buffer into Encapsulated PostScript.
//
// Use: public

SbBool
InvOffscreenRenderer::writeToPostScript(const char *filename) const
//
////////////////////////////////////////////////////////////////////////
{
    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        cerr << " Could not open file " << filename << endl;
        cerr << "InvOffscreenRenderer::writeToPostScript(.-1-)" << endl;
        return FALSE;
    }

    const SbVec2s &vpSize = renderedViewport.getViewportSizePixels();
    SbVec2f printSize;
    float ppi = renderedViewport.getPixelsPerInch();

    printSize[0] = vpSize[0] / ppi;
    printSize[1] = vpSize[1] / ppi;

    return (writeToPostScript(filename, printSize));
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Write the buffer into Encapsulated PostScript at the given
//    print size in inches.
//
// Use: public

SbBool
InvOffscreenRenderer::writeToPostScript(
    const char *filename,
    const SbVec2f &printSize) const

//
////////////////////////////////////////////////////////////////////////
{
    FILE *fp = fopen(filename, "w");
    if (!fp)
    {
        cerr << " Could not open file " << filename << endl;
        cerr << "InvOffscreenRenderer::writeToPostScript(.-2-)" << endl;
        return FALSE;
    }

    // Set the graphics context to be the offscreen pixmap
    if (!setContext())
        return FALSE;

    int components;
    if ((comps == LUMINANCE_TRANSPARENCY) || (comps == RGB_TRANSPARENCY))
    {
#ifdef DEBUG
        SoDebugError::post("InvOffscreenRenderer::writeToRGB",
                           "PostScript does not support image transparency");
#endif
        return FALSE;
    }
    else if (comps == InvOffscreenRenderer::LUMINANCE)
    {
        components = 1;
    }
    else
    {
        components = 3;
    }

    // Get the correct viewport region
    const SbVec2s &vpSize = renderedViewport.getViewportSizePixels();

    // Write the PostScript header
    fprintf(fp, "%%!PS-Adobe-2.0 EPSF-1.2\n");
    fprintf(fp, "%%%%Creator: IRIS program output\n");
    fprintf(fp, "%%%%BoundingBox: 0 0 %d %d\n", vpSize[0], vpSize[1]);
    fprintf(fp, "%%%%EndComments\n");
    fprintf(fp, "gsave\n");

    // Write the image into the PostScript file
    fprintf(fp, "/bwproc {\n");
    fprintf(fp, "    rgbproc\n");
    fprintf(fp, "    dup length 3 idiv string 0 3 0\n");
    fprintf(fp, "    5 -1 roll {\n");
    fprintf(fp, "    add 2 1 roll 1 sub dup 0 eq\n");
    fprintf(fp, "    { pop 3 idiv 3 -1 roll dup 4 -1 roll dup\n");
    fprintf(fp, "        3 1 roll 5 -1 roll put 1 add 3 0 }\n");
    fprintf(fp, "    { 2 1 roll } ifelse\n");
    fprintf(fp, "    } forall\n");
    fprintf(fp, "    pop pop pop\n");
    fprintf(fp, "} def\n");
    fprintf(fp, "systemdict /colorimage known not {\n");
    fprintf(fp, "    /colorimage {\n");
    fprintf(fp, "        pop\n");
    fprintf(fp, "        pop\n");
    fprintf(fp, "        /rgbproc exch def\n");
    fprintf(fp, "        { bwproc } image\n");
    fprintf(fp, "    } def\n");
    fprintf(fp, "} if\n");
    fprintf(fp, "/picstr %d string def\n", vpSize[0] * components);
    fprintf(fp, "%d %d scale\n", (int)(printSize[0] * 72.0),
            (int)(printSize[1] * 72.0));
    fprintf(fp, "%d %d %d\n", vpSize[0], vpSize[1], 8);
    fprintf(fp, "[%d 0 0 %d 0 0]\n", vpSize[0], vpSize[1]);
    fprintf(fp, "{currentfile picstr readhexstring pop}\n");
    fprintf(fp, "false %d\n", components);
    fprintf(fp, "colorimage\n");

    // Get the format of the pixmap data
    GLenum format;
    getFormat(format);

    // Convert the pixel values to ASCII hex and write them out.
    unsigned char *pBuf = new unsigned char[vpSize[0] * components * 2];

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    int numValues = vpSize[0] * components;
    int hexpos = 0;

    for (int row = 0; row < vpSize[1]; row++)
    {

        // Read the next scanline of pixels from the offscreen pixmap
        glReadPixels(0, row, vpSize[0], 1, format,
                     GL_UNSIGNED_BYTE, (GLvoid *)pBuf);

        char *tbuf = (char *)pBuf;

        // Write out the scanline
        for (int i = 0; i < numValues; i++)
            putHex(fp, (char)*tbuf++, hexpos);
    }

    if (hexpos)
        fprintf(fp, "\n");

    // Finish up the PostScript file.
    fprintf(fp, "grestore\n");
    fprintf(fp, "showpage\n");

    delete[] pBuf;
    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Setup a pixmap into which the rendering will occur.
//
// Use: private

SbBool
InvOffscreenRenderer::setupPixmap()

//
////////////////////////////////////////////////////////////////////////
{
    const SbVec2s &vpSize = renderedViewport.getViewportSizePixels();

    if (!initPixmap(display, visual, context, vpSize, pixmap, pmap))
    {
        return FALSE;
    }

    // Clear the pixmap to the backgroundColor
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glClearColor(backgroundColor[0], backgroundColor[1], backgroundColor[2], 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Setup a pixmap into which the rendering will occur.
//
// Use: static, private

SbBool
InvOffscreenRenderer::initPixmap(
    Display *&dpy,
    XVisualInfo *&vi,
    GLXContext &cx,
    const SbVec2s &sz,
    GLXPixmap &glxpmap,
    Pixmap &xpmap)

//
////////////////////////////////////////////////////////////////////////
{

    // If a display is already open, and is the same size, use it.
    // Otherwise, close it and open up a fresh one.
    if (dpy != NULL)
    {

        // Get the size of the passed pixmap.  If it is the same size
        // as the passed size, just return.
        Window root;
        int x, y;
        unsigned int width, height, border_width, depth;
        if (!XGetGeometry(dpy, xpmap, &root, &x, &y, &width, &height,
                          &border_width, &depth))
        {

            // Bad pixmap.  Destroy it.
            glXDestroyGLXPixmap(dpy, glxpmap);
            glXDestroyContext(dpy, cx);
            XCloseDisplay(dpy);
        }
        else if ((width == (unsigned int)sz[0]) && (height == (unsigned int)sz[1]))
        {

            // This pixmap is the right size to use for this rendering.
            // Make it the current context for rendering.
            if (!glXMakeCurrent(dpy, glxpmap, cx))
            {
                glXDestroyGLXPixmap(dpy, glxpmap);
                glXDestroyContext(dpy, cx);
                XCloseDisplay(dpy);
                dpy = NULL;
                return FALSE;
            }
            // Store the pixels as byte aligned rows in the Pixmap.
            glPixelStorei(GL_PACK_ALIGNMENT, 1);

            return TRUE;
        }
        else
        {
            // The pixmap is not the right size.  Destroy it.
            glXDestroyGLXPixmap(dpy, glxpmap);
            glXDestroyContext(dpy, cx);
            XCloseDisplay(dpy);
        }
    }

    // Create an X Display
    if ((dpy = XOpenDisplay(NULL)) == NULL)
        return FALSE;

    // Create a GLX Visual
    if ((vi = glXChooseVisual(dpy, DefaultScreen(dpy),
                              attributeList)) == NULL)
    {

        cerr << "InvOffscreenRenderer::initPixmap(..) failed in line " << __LINE__
             << " glXChooseVisual(..) " << endl;

        //        XCloseDisplay(dpy);
        dpy = NULL;
        return FALSE;
    }

    if (vi->depth < 24)
    {
        cerr << "offscreen renderering only possible for color depths greater equal 24 " << endl;
        dpy = NULL;
        return FALSE;
    }

    // Create a GLX Context
    if ((cx = glXCreateContext(dpy, vi, NULL, FALSE)) == NULL)
    {
        //        XCloseDisplay(dpy);
        dpy = NULL;
        return FALSE;
    }

    // Create X and GLX Pixmap
    xpmap = XCreatePixmap(dpy, DefaultRootWindow(dpy),
                          (unsigned int)sz[0], (unsigned int)sz[1], vi->depth);
    glxpmap = glXCreateGLXPixmap(dpy, vi, xpmap);

    if (!glXMakeCurrent(dpy, glxpmap, cx))
    {
        glXDestroyGLXPixmap(dpy, glxpmap);
        glXDestroyContext(dpy, cx);
        XCloseDisplay(dpy);
        dpy = NULL;
        return FALSE;
    }

    // Store the pixels as byte aligned rows in the Pixmap
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Read the pixels from the Pixmap
//
// Use: private

void
InvOffscreenRenderer::readPixels()

//
////////////////////////////////////////////////////////////////////////
{
    const SbVec2s &vpSize = renderedViewport.getViewportSizePixels();

    GLenum format;
    int allocSize;
    switch (comps)
    {
    case LUMINANCE:
        format = GL_LUMINANCE;
        allocSize = vpSize[0] * vpSize[1] * 1;
        break;
    case LUMINANCE_TRANSPARENCY:
        format = GL_LUMINANCE_ALPHA;
        allocSize = vpSize[0] * vpSize[1] * 2;
        break;
    case RGB:
        format = GL_RGB;
        allocSize = vpSize[0] * vpSize[1] * 3;
        break;
    case RGB_TRANSPARENCY:
        format = GL_RGBA;
        allocSize = vpSize[0] * vpSize[1] * 4;
        break;
    default:
        format = GL_LUMINANCE;
        allocSize = vpSize[0] * vpSize[1] * 1;
        fprintf(stderr, "InvOffscreenRenderer::readPixels(): unhandled format\n");
        break;
    }

    pixelBuffer = new unsigned char[allocSize];
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glReadPixels(0, 0, vpSize[0], vpSize[1], format,
                 GL_UNSIGNED_BYTE, (GLvoid *)pixelBuffer);
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Set the graphics context to point to the offscreen pixmap
//
// Use: private

SbBool
InvOffscreenRenderer::setContext() const

//
////////////////////////////////////////////////////////////////////////
{
    if (!glXMakeCurrent(display, pixmap, context))
    {
        glXDestroyGLXPixmap(display, pixmap);
        glXDestroyContext(display, context);
        XCloseDisplay(display);
        return FALSE;
    }

    return TRUE;
}

////////////////////////////////////////////////////////////////////////
//
// Description:
//    Get format information
//
// Use: private

void
InvOffscreenRenderer::getFormat(GLenum &format) const

//
////////////////////////////////////////////////////////////////////////
{
    switch (comps)
    {
    case LUMINANCE:
        format = GL_LUMINANCE;
        break;
    case LUMINANCE_TRANSPARENCY:
        format = GL_LUMINANCE_ALPHA;
        break;
    case RGB:
        format = GL_RGB;
        break;
    case RGB_TRANSPARENCY:
        format = GL_RGBA;
        break;
    default:
        cerr << "unhandled number of components, assuming GL_RGBA pixel format" << endl;
        format = GL_RGBA;
        break;
    }
}

SbBool
InvOffscreenRenderer::writeToTiff(const char *fileName) const
{
    // Set the graphics context to be the offscreen pixmap
    if (!setContext())
        return FALSE;

    // Open an image file for writing
    int dimensions, components;
    if (comps == (InvOffscreenRenderer::LUMINANCE_TRANSPARENCY) || (comps == InvOffscreenRenderer::RGB_TRANSPARENCY))
    {
#ifdef DEBUG
        SoDebugError::post("InvOffscreenRenderer::writeToTiff",
                           "we do not support transparency yet");
#endif
        return FALSE;
    }
    else if (comps == InvOffscreenRenderer::LUMINANCE)
    {
        dimensions = 2;
        components = 1;
    }
    else
    {
        dimensions = 3;
        components = 3;
    }

    // Get the correct viewport size.
    const SbVec2s &vpSize = renderedViewport.getViewportSizePixels();

    const int buflen(components * vpSize[0] * vpSize[1]);
    unsigned char *tmpbuffer = new unsigned char[buflen];
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, vpSize[0], vpSize[1], GL_RGB,
                 GL_UNSIGNED_BYTE, tmpbuffer);

    char *buffer = new char[buflen];
    // strange border width
    int di(0);

    int i, j, idx;
    int mBufLen(3 * (vpSize[0] - di) * (vpSize[1] - di));
    int cnt(0);
    for (i = 0; i < vpSize[1] - di; ++i)
    {
        for (j = di; j < vpSize[0]; j++)
        {
            idx = 3 * ((i + 1) * vpSize[0] - j);
            buffer[cnt] = (char)tmpbuffer[buflen - idx];
            buffer[cnt + 1] = (char)tmpbuffer[buflen + 1 - idx];
            buffer[cnt + 2] = (char)tmpbuffer[buflen + 2 - idx];
            cnt += 3;
        }
    }
    // write tiff file

    TIFF *image;

    // Open the TIFF file
    if ((image = TIFFOpen(fileName, "w")) == NULL)
    {
#ifdef DEBUG
        SoDebugError::post("InvOffscreenRenderer::writeToTiff",
                           "Unable to open tiff file");
#endif
        return FALSE;
    }

    // We need to set some values for basic tags before we can add any data
    TIFFSetField(image, TIFFTAG_IMAGEWIDTH, vpSize[0] - di);
    TIFFSetField(image, TIFFTAG_IMAGELENGTH, vpSize[1] - di);
    TIFFSetField(image, TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(image, TIFFTAG_SAMPLESPERPIXEL, dimensions);
    TIFFSetField(image, TIFFTAG_ROWSPERSTRIP, vpSize[1] - di);

    TIFFSetField(image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    TIFFSetField(image, TIFFTAG_FILLORDER, FILLORDER_MSB2LSB);
    TIFFSetField(image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);

    TIFFSetField(image, TIFFTAG_XRESOLUTION, 150.0);
    TIFFSetField(image, TIFFTAG_YRESOLUTION, 150.0);
    TIFFSetField(image, TIFFTAG_RESOLUTIONUNIT, RESUNIT_INCH);

    // Write the information to the file
    TIFFWriteEncodedStrip(image, 0, buffer, mBufLen);

    // Close the file
    TIFFClose(image);

    delete[] tmpbuffer;
    delete[] buffer;

    return TRUE;
}
