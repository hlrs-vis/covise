/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                           (C)2007 USAL  **
 **                                                                         **
 ** Description: 3D Video                                                   **
 **                                                                         **
 **                                                                         **
 ** Author: Dulcidio Coelho		                                            **
 **                                                                         **
 ** History:  								                                **
 ** MAY-07  v1	    				       		                            **
 **                                                                         **
 **                                                                         **
\****************************************************************************/

#include "3DVideo.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>

Video3DNode::Video3DNode()
{
    setSupportsDisplayList(false);
}

Video3DNode::~Video3DNode()
{
}

/** Clone the type of an object, with Object* return type.
Must be defined by derived classes.*/
osg::Object *Video3DNode::cloneType() const
{
    return new Video3DNode();
}

/** Clone the an object, with Object* return type.
Must be defined by derived classes.*/
osg::Object *Video3DNode::clone(const osg::CopyOp &) const
{
    return new Video3DNode();
}

void Video3DNode::drawImplementation(osg::RenderInfo &renderInfo) const
{
    bool rightVideo = false;
    if (osg::View *view = renderInfo.getView())
    {
        if (osg::State *state = renderInfo.getState())
        {
            if (const osg::DisplaySettings *ds = state->getDisplaySettings())
            {
                switch (ds->getStereoMode())
                {
                case osg::DisplaySettings::HORIZONTAL_INTERLACE:
                case osg::DisplaySettings::VERTICAL_INTERLACE:
                case osg::DisplaySettings::ANAGLYPHIC:
                    /* TODO */
                    break;
                case osg::DisplaySettings::HORIZONTAL_SPLIT:
                case osg::DisplaySettings::VERTICAL_SPLIT:
                    if (osg::Camera *cam = view->getCamera())
                    {
                        for (int i = 0; i < cover->numScreens; ++i)
                        {
                            if (cover->screens[i].camera.get() == cam)
                            {
                                rightVideo = cover->screens[i].stereoMode == osg::DisplaySettings::RIGHT_EYE;
                                break;
                            }
                        }
                    }
                    break;
                case osg::DisplaySettings::LEFT_EYE:
                    break;
                case osg::DisplaySettings::RIGHT_EYE:
                    rightVideo = true;
                    break;
                case osg::DisplaySettings::QUAD_BUFFER:
                    if (osg::Camera *cam = view->getCamera())
                    {
                        rightVideo = (cam->getDrawBuffer() == GL_BACK_RIGHT || cam->getDrawBuffer() == GL_FRONT_RIGHT);
                    }
                    break;
                default:
                    cerr << "Video3DNode::drawImplementation: unknown stereo mode" << endl;
                    break;
                }
            }
        }
    }

    // add OpenGL code here

    //osg::StateSet * currentState = new osg::StateSet;
    //renderInfo.getState()->captureCurrentState(*currentState);
    //renderInfo.getState()->pushStateSet(currentState);

    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();

    if (rightVideo)
    {
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_QUADS); // Draw A Quad
        glVertex3f(-1.0f, -1.0f, 1.0f); // Bottom Left
        glVertex3f(1.0f, -1.0f, 1.0f); // Bottom Right
        glVertex3f(1.0f, 1.0f, 1.0f); // Top Right
        glVertex3f(-1.0f, 1.0f, 1.0f); // Top Left
        glEnd();
        glBegin(GL_QUADS); // Draw A Quad
        glVertex3f(-1.0f, -1.0f, 1.0f); // Bottom Left
        glVertex3f(-1.0f, 1.0f, 1.0f); // Top Left
        glVertex3f(1.0f, 1.0f, 1.0f); // Top Right
        glVertex3f(1.0f, -1.0f, 1.0f); // Bottom Right
        glEnd();
    }
    else
    {
        glColor3f(1.0f, 0.0f, 0.0f);
        glTranslatef(3.0f, 0.0f, 0.0f); // Move Right 3 Units
        glBegin(GL_QUADS); // Draw A Quad
        glVertex3f(-1.0f, -1.0f, 1.0f); // Bottom Left
        glVertex3f(1.0f, -1.0f, 1.0f); // Bottom Right
        glVertex3f(1.0f, 1.0f, 1.0f); // Top Right
        glVertex3f(-1.0f, 1.0f, 1.0f); // Top Left
        glEnd();
        glBegin(GL_QUADS); // Draw A Quad
        glVertex3f(-1.0f, -1.0f, 1.0f); // Bottom Left
        glVertex3f(-1.0f, 1.0f, 1.0f); // Top Left
        glVertex3f(1.0f, 1.0f, 1.0f); // Top Right
        glVertex3f(1.0f, -1.0f, 1.0f); // Bottom Right
        glEnd();
    }

    //renderInfo.getState()->popStateSet();
}

osg::BoundingBox Video3DNode::computeBound() const
{
    return osg::BoundingBox(-5.0f, -5.0f, -5.0f, 5.0f, 5.0f, 5.0f);
}

Video3D::Video3D()
{
    fprintf(stderr, "Video3D::Video3D\n");
    videoNode = new Video3DNode();
    geodevideo = new osg::Geode;
    geodevideo->addDrawable(videoNode);
    cover->getObjectsRoot()->addChild(geodevideo);
}

// this is called if the plugin is removed at runtime
Video3D::~Video3D()
{
    fprintf(stderr, "Video3D::~Video3D\n");
    cover->getObjectsRoot()->removeChild(geodevideo);
}

// here we get the size and the current center of the cube
void
Video3D::feedback(coInteractor *i)
{
    (void)i;
    fprintf(stderr, "Video3D::feedback\n");
}

void Video3D::addObject(RenderObject *container,
                        RenderObject *obj, RenderObject *normObj,
                        RenderObject *colorObj, RenderObject *texObj,
                        const char *root,
                        int numCol, int colorBinding, int colorPacking,
                        float *r, float *g, float *b, int *packedCol,
                        int numNormals, int normalBinding,
                        float *xn, float *yn, float *zn, float transparency)
{
    (void)container;
    (void)obj;
    (void)normObj;
    (void)colorObj;
    (void)texObj;
    (void)root;
    (void)numCol;
    (void)colorBinding;
    (void)colorPacking;
    (void)r;
    (void)g;
    (void)b;
    (void)packedCol;
    (void)numNormals;
    (void)normalBinding;
    (void)xn;
    (void)yn;
    (void)zn;
    (void)transparency;
    fprintf(stderr, "Video3D::addObject\n");
}

void
Video3D::removeObject(const char *objName, bool replace)
{
    (void)objName;
    (void)replace;
    fprintf(stderr, "Video3D::removeObject\n");
}

void
Video3D::preFrame()
{
}

// C plugin interface, don't do any coding down here, do it in the C++ Class!

Video3D *plugin = NULL;

int coVRInit(coVRPlugin *m)
{
    (void)m;
    plugin = new Video3D();
    if (plugin)
        return (1);
    else
        return (0);
}

void coVRDelete(coVRPlugin *m)
{
    (void)m;
    delete plugin;
}

void coVRPreFrame()
{
    plugin->preFrame();
}

void coVRNewInteractor(RenderObject * /*container*/, coInteractor *i)
{
    plugin->feedback(i);
}

void coVRAddObject(RenderObject *container,
                   RenderObject *obj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   const char *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn,
                   float transparency)
{
    plugin->addObject(container, obj, normObj, colorObj, texObj, root, numCol, colorBinding, colorPacking,
                      r, g, b, packedCol, numNormals, normalBinding,
                      xn, yn, zn, transparency);
}

void coVRRemoveObject(const char *objName, int replace)
{
    plugin->removeObject(objName, replace);
}
