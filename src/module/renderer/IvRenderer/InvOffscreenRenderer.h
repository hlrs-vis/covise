/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    InvOffscreenRenderer
//
// Description: offscreen renderer for Inventor based on SGI's offscreen renderer
//              contains possibility to create tiff files and general rgb's
//
// Initial version: 21.10.2002
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef INVOFFCREENRENDERER_H
#define INVOFFCREENRENDERER_H

#include <Inventor/SbLinear.h>
#include <Inventor/SbColor.h>
#include <Inventor/SoPath.h>
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/errors/SoDebugError.h>
#include <Inventor/nodes/SoNode.h>

#include <GL/glx.h>

// SGI's original header:
//////////////////////////////////////////////////////////////////////////////
//
//  Class: SoOffscreenRenderer
//
//  This file contains the definition of the SoOffscreenRenderer class.
//  This class is used for rendering a scene graph to an offscreen memory
//  buffer which can be used for printing or generating textures.
//
//  The implementation of this class uses the X Pixmap for rendering.
//
//////////////////////////////////////////////////////////////////////////////

// C-api: prefix=SoOffRnd

class InvOffscreenRenderer
{
public:
    // Constructor
    InvOffscreenRenderer(const SbViewportRegion &viewportRegion);
    // C-api: name=CreateAct
    InvOffscreenRenderer(SoGLRenderAction *ra);

    // Destructor
    ~InvOffscreenRenderer();

    enum Components
    {
        LUMINANCE = 1,
        LUMINANCE_TRANSPARENCY = 2,
        RGB = 3, // The default
        RGB_TRANSPARENCY = 4
    };

    static float getScreenPixelsPerInch();

    // Set/get the components to be rendered
    void setComponents(Components components)
    {
        comps = components;
    }

    Components getComponents() const
    {
        return comps;
    }

    // Set/get the viewport region
    void setViewportRegion(const SbViewportRegion &region);

    const SbViewportRegion &getViewportRegion() const;

    // Get the maximum supported resolution of the viewport.
    static SbVec2s getMaximumResolution();

    // Set/get the background color
    void setBackgroundColor(const SbColor &c)
    {
        backgroundColor = c;
    }

    const SbColor &getBackgroundColor() const
    {
        return backgroundColor;
    }

    // Set and get the render action to use
    void setGLRenderAction(SoGLRenderAction *ra);

    SoGLRenderAction *getGLRenderAction() const;

    // Render the given scene into a buffer
    SbBool render(SoNode *scene);

    SbBool render(SoPath *scene);

    // Return the buffer containing the rendering
    unsigned char *getBuffer() const;

    // Write the buffer as a .rgb file into the given FILE
    SbBool writeToRGB(const char *filename) const;

    // Write the buffer into encapsulated PostScript.  If a print size is
    // not given, adjust the size of the print so it is WYSIWYG with respect
    // to the viewport region on the current device.
    SbBool writeToPostScript(const char *filename) const;

    SbBool writeToPostScript(const char *filename,
                             const SbVec2f &printSize) const;

    SbBool writeToTiff(const char *fileName) const;

private:
    unsigned char *pixelBuffer;
    Components comps;
    SbColor backgroundColor;
    SoGLRenderAction *userAction, *offAction;
    SbViewportRegion renderedViewport;

    // These are used for rendering to the offscreen pixmap
    Display *display;
    XVisualInfo *visual;
    GLXContext context;
    GLXPixmap pixmap;
    Pixmap pmap;

    // Setup the offscreen pixmap
    SbBool setupPixmap();

    // Initialize an offscreen pixmap
    static SbBool initPixmap(Display *&dpy, XVisualInfo *&vi,
                             GLXContext &cx, const SbVec2s &sz,
                             GLXPixmap &glxPmap, Pixmap &xpmap);

    // Read pixels back from the Pixmap
    void readPixels();

    // Set the graphics context
    SbBool setContext() const;

    // Return the format used in the rendering
    void getFormat(GLenum &format) const;

    static void putHex(FILE *fp, char val, int &hexPos);
};
#endif
