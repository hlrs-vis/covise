/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cassert>
#include <iostream>
#include <ostream>

#ifdef COVER
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#endif

// OSG:
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osgUtil/SceneView>

// Local:
#include "Interface.h"

using std::cerr;
using std::endl;
using namespace osg;
using namespace cui;
using namespace covise;
using namespace opencover;

Interface::Interface()
{
    Matrix rotX, rotY, rotZ, trans, scale, mat;

    setOSGLibraryPath();

    // Build the top of the scene graph, which is the virtual world:
    worldRoot = new ClearNode();
    worldRoot->setRequiresClear(false); // don't clear frame buffer - this is done by host application

    // Build the scene graph node for the Cave room:
    _r2w.makeIdentity();
    roomRoot = new MatrixTransform(_r2w);

    // Build the scene graph node for the head coordinates:
    _h2r.makeIdentity();
    headRoot = new MatrixTransform(_h2r);

    // Build the scene graph node for left wall coordinates:
    rotX.makeRotate(M_PI, 1, 0, 0);
    rotY.makeRotate(-M_PI / 2.0, 0, 1, 0);
    trans.makeTranslate(-4, 4, -4);
    mat = rotX * rotY * trans;
    leftRoot = new MatrixTransform(mat);

    // Build the scene graph node for right wall coordinates:
    rotX.makeRotate(M_PI, 1, 0, 0);
    rotY.makeRotate(M_PI / 2.0, 0, 1, 0);
    trans.makeTranslate(4, 4, 4);
    mat = rotX * rotY * trans;
    rightRoot = new MatrixTransform(mat);

    // Build the scene graph node for right Rave wall coordinates:
    rotX.makeRotate(M_PI, 1, 0, 0);
    rotY.makeRotate(M_PI / 2.0, 0, 1, 0);
    trans.makeTranslate(10, 5, 5);
    mat = rotX * rotY * trans;
    rightRaveRoot = new MatrixTransform(mat);

    // Build the scene graph node for front wall coordinates:
    rotX.makeRotate(M_PI, 1, 0, 0);
    trans.makeTranslate(-4, 4, 4);
    mat = rotX * trans;
    frontRoot = new MatrixTransform(mat);

    // Build the scene graph node for floor coordinates:
    rotX.makeRotate(M_PI / 2.0, 1, 0, 0);
    trans.makeTranslate(-4, 4, -4);
    mat = rotX * trans;
    floorRoot = new MatrixTransform(mat);

    // Build the scene graph node for fishtank coordinates:
    rotX.makeRotate(M_PI, 1, 0, 0);
    trans.makeTranslate(0.7, 0.5, 4); // normal setting: GUI on the right monitor of a dual-head system
    //  trans.makeTranslate(-0.6, 0.5, 4); // GUI on left monitor, or main monitor if single-head
    scale.makeScale(1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0);
    mat = rotX * scale * trans;
    fishRoot = new MatrixTransform(mat);

    // Create the Scene View
    sceneview = new osgUtil::SceneView();
    sceneview->setDefaults();
    sceneview->setSceneData(worldRoot.get());
    // let Inspace manage the z buffer
    sceneview->setComputeNearFarMode(osgUtil::CullVisitor::DO_NOT_COMPUTE_NEAR_FAR);
    worldRoot->addChild(roomRoot.get());
    roomRoot->addChild(headRoot.get());
    roomRoot->addChild(leftRoot.get());
    roomRoot->addChild(rightRoot.get());
    roomRoot->addChild(rightRaveRoot.get());
    roomRoot->addChild(frontRoot.get());
    roomRoot->addChild(floorRoot.get());
    roomRoot->addChild(fishRoot.get());

    // Initialize display type:
    std::string displayDevice = coCoviseConfig::getEntry("OSGCaveUI.DisplayDevice");
    if (displayDevice.empty())
    {
        cerr << "Error: config variable DisplayDevice not set." << endl;
        cerr << "Using CAVE by default." << endl;
    }

    if (displayDevice == "desktopdualhead")
        _display = FISHTANK;
    else if (displayDevice == "desktop")
        _display = DESKTOP;
    else
        _display = CAVE;

    cerr << "Display type is " << _display << endl;
}

Interface::~Interface()
{
    sceneview.release();
    worldRoot.release();
    roomRoot.release();
    leftRoot.release();
    rightRoot.release();
    rightRaveRoot.release();
    frontRoot.release();
    floorRoot.release();
    fishRoot.release();
    headRoot.release();
}

/** Set environment variable OSG_LD_LIBRARY_PATH to default $G OSG path,
  if it is not defined. Setting the user environment variables does not
  work in the Cave.
*/
void Interface::setOSGLibraryPath()
{
    const char *libPath = "/lib/osg";
    const char *pluginsPath = "/lib/osg/osgPlugins";
    const char *filePath = "/lib/osg";
    const char *varLibPath = "OSG_LD_LIBRARY_PATH";
    const char *varFilePath = "OSG_FILE_PATH";
    const char *varGPath = "G";
    const char *dollarGPath = "/share/gfx/tools/linux";
    const char *varExtDisable = "OSG_GL_EXTENSION_DISABLE";
    const char *extDisable = "GL_SGIS_generate_mipmap";
    const char *dollarG;

    if (getenv(varGPath))
    {
        cerr << varGPath << "=" << getenv(varGPath) << endl;
    }
    else
    {
        char *envStr1 = new char[strlen(varGPath) + 1 + strlen(dollarGPath) + 1];
        sprintf(envStr1, "%s=%s", varGPath, dollarGPath);
        putenv(envStr1);
        cerr << varGPath << " has been set to " << getenv(varGPath) << endl;
        // don't delete env, will be taken care of by putenv
    }

    dollarG = getenv(varGPath);
    assert(dollarG);

    if (getenv(varLibPath))
    {
        cerr << varLibPath << "=" << getenv(varLibPath) << endl;
    }
    else
    {
        char *envStr2 = new char[strlen(varLibPath) + 1 + strlen(dollarG) + strlen(libPath) + 1 + strlen(dollarG) + strlen(pluginsPath) + 1];
        sprintf(envStr2, "%s=%s%s:%s%s", varLibPath, dollarG, libPath, dollarG, pluginsPath);
        putenv(envStr2);
        cerr << varLibPath << " has been set to " << getenv(varLibPath) << endl;
        // don't delete envStr, will be taken care of by putenv
    }

    if (getenv(varFilePath))
    {
        cerr << varFilePath << "=" << getenv(varFilePath) << endl;
    }
    else
    {
        char *envStr3 = new char[strlen(varFilePath) + 1 + strlen(dollarG) + strlen(filePath) + 1];
        sprintf(envStr3, "%s=%s%s", varFilePath, dollarG, filePath);
        putenv(envStr3);
        cerr << varFilePath << " has been set to " << getenv(varFilePath) << endl;
        // don't delete envStr, will be taken care of by putenv
    }

    if (getenv(varExtDisable))
    {
        cerr << varExtDisable << "=" << getenv(varExtDisable) << endl;
    }
    else
    {
        char *envStr4 = new char[strlen(varExtDisable) + 1 + strlen(extDisable) + 1];
        sprintf(envStr4, "%s=%s", varExtDisable, extDisable);
        putenv(envStr4);
        cerr << varExtDisable << " has been set to " << getenv(varExtDisable) << endl;
        // don't delete envStr, will be taken care of by putenv
    }
}

/** Returns the scenegraph's virtual world root node.
  Attach objects that are fixed relative to the virtual world.
*/
ref_ptr<ClearNode> Interface::getWorldRoot()
{
    return worldRoot;
}

/** Returns the scenegraph's real world (eg, Cave) root node.
  Attach objects that are fixed relative to the physical Cave.
*/
ref_ptr<MatrixTransform> Interface::getRoomRoot()
{
    return roomRoot;
}

/** Returns the scenegraph's head coordinates. The origin of
  this coordinate system is between the viewer's eyes.
  Attach objects that are fixed relative to the head.
*/
ref_ptr<MatrixTransform> Interface::getHeadRoot()
{
    return headRoot;
}

/** Returns the root node to the left wall coordinate system.
  The origin of this coordinate system is at the bottom left corner
  of the left cave wall; x is to the right, y is up.
*/
ref_ptr<MatrixTransform> Interface::getLeftRoot()
{
    return leftRoot;
}

/** Returns the root node to the right wall coordinate system.
  The origin of this coordinate system is at the bottom left corner
  of the right cave wall; x is to the right, y is up.
*/
ref_ptr<MatrixTransform> Interface::getRightRoot()
{
    return rightRoot;
}

ref_ptr<MatrixTransform> Interface::getRightRaveRoot()
{
    return rightRaveRoot;
}

/** Returns the root node to the front wall coordinate system.
  The origin of this coordinate system is at the bottom left corner
  of the front cave wall; x is to the right, y is up.
*/
ref_ptr<MatrixTransform> Interface::getFrontRoot()
{
    return frontRoot;
}

/** Returns the root node to the Cave floor coordinate system.
  The origin of this coordinate system is at the bottom left corner
  of the Cave floor; x is to the right, y is up.
*/
ref_ptr<MatrixTransform> Interface::getFloorRoot()
{
    return floorRoot;
}

/** Returns the root node to the Cave fishtank coordinate system.
  The origin of this coordinate system is at the bottom left corner
  of the right monitor; x is to the right, y is up.
*/
ref_ptr<MatrixTransform> Interface::getFishRoot()
{
    return fishRoot;
}

/** Returns the SceneView object.
 */
ref_ptr<osgUtil::SceneView> Interface::getSceneView()
{
    return sceneview;
}

/** This is a more elegant way for:
  d->getWorldRoot().get()->addChild(node);
*/
void Interface::addWorldChild(Node *node)
{
    worldRoot->addChild(node);
}

void Interface::removeWorldChild(Node *node)
{
    worldRoot->removeChild(node);
}

/** This is a more elegant way for:
  d->getRoomRoot().get()->addChild(node);
*/
void Interface::addRoomChild(Node *node)
{
    roomRoot->addChild(node);
}

void Interface::removeRoomChild(Node *node)
{
    roomRoot->removeChild(node);
}

/** This is a more elegant way for:
  d->getHeadRoot().get()->addChild(node);
*/
void Interface::addHeadChild(Node *node)
{
    headRoot->addChild(node);
}

void Interface::removeHeadChild(Node *node)
{
    headRoot->removeChild(node);
}

/** This is a more elegant way for:
  d->getLeftRoot().get()->addChild(node);
*/
void Interface::addLeftChild(Node *node)
{
    leftRoot->addChild(node);
}

void Interface::removeLeftChild(Node *node)
{
    leftRoot->removeChild(node);
}

/** This is a more elegant way for:
  d->getRightRoot().get()->addChild(node);
*/
void Interface::addRightChild(Node *node)
{
    rightRoot->addChild(node);
}

void Interface::addRightRaveChild(Node *node)
{
    rightRaveRoot->addChild(node);
}

void Interface::removeRightChild(Node *node)
{
    rightRoot->removeChild(node);
}

void Interface::removeRightRaveChild(Node *node)
{
    rightRaveRoot->removeChild(node);
}

/** This is a more elegant way for:
  d->getFrontRoot().get()->addChild(node);
*/
void Interface::addFrontChild(Node *node)
{
    frontRoot->addChild(node);
}

void Interface::removeFrontChild(Node *node)
{
    frontRoot->removeChild(node);
}

/** This is a more elegant way for:
  d->getFloorRoot().get()->addChild(node);
*/
void Interface::addFloorChild(Node *node)
{
    floorRoot->addChild(node);
}

/** This is a more elegant way for:
  d->getFishRoot().get()->addChild(node);
*/
void Interface::addFishChild(Node *node)
{
    fishRoot->addChild(node);
}

void Interface::removeFloorChild(Node *node)
{
    floorRoot->removeChild(node);
}

void Interface::removeFishChild(Node *node)
{
    fishRoot->removeChild(node);
}

/** Enable/disable culling of small objects.
  Default: enabled
*/
void Interface::setSmallFeatureCulling(bool turnOn)
{
    CullStack::CullingMode cullingMode = sceneview->getCullingMode();
    if (turnOn)
    {
        cullingMode |= CullStack::SMALL_FEATURE_CULLING;
    }
    else
    {
        cullingMode &= ~(CullStack::SMALL_FEATURE_CULLING);
    }
    sceneview->setCullingMode(cullingMode);
}

void Interface::draw()
{
    Matrix projMat;
    Matrix viewMat;
    GLfloat glmatrix[16];
    GLint view[4];

// Update matrices:
#ifdef DOLLAR_G
    _r2w.set(WorldTranslate::instance()->ROOM_TO_WORLD().matrix());
    _h2r.set(ISVREngine::instance()->headXform().matrix());
#elif COVER
    Matrix h2r = cover->getViewerMat();
    Matrix r2w = cover->getXformMat();
    _r2w.set(r2w);
    _h2r.set(h2r);
#endif

    _h2w = _r2w * _h2r;
    _w2h.invert(_h2w);
    _w2r.invert(_r2w);
    _r2h.invert(_h2r);

    // Save OpenGL state:
    glPushAttrib(GL_ALL_ATTRIB_BITS); // TODO: only save necessary bits if this takes too long
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();

    // Set OSG viewport:
    glGetIntegerv(GL_VIEWPORT, view);
    sceneview->setViewport(view[0], view[1], view[2], view[3]);

    // Set OSG projection matrix:
    glGetFloatv(GL_PROJECTION_MATRIX, glmatrix);
    projMat.set((float *)glmatrix);
    sceneview->setProjectionMatrix(projMat);

    // Set OSG modelview matrix:
    glGetFloatv(GL_MODELVIEW_MATRIX, glmatrix);
    viewMat.set((float *)glmatrix);
    sceneview->setViewMatrix(viewMat);

    // Update room node matrix:
    roomRoot->setMatrix(_r2w);

    // Update head matrix:
    headRoot->setMatrix(_h2r);

    // z-buffer settings
    if (_zbuf_flag) // XXX - not sure this works...
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);

    // Call OSG drawing routines:
    sceneview->update();
    sceneview->cull();
    sceneview->draw();

    // Restore OpenGL state:
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glPopAttrib();
}

/** @return true if child is a child of parent. This works over
  multiple levels of inheritance!
*/
bool Interface::isChild(Node *child, Node *parent)
{
    Node *testParent;
    for (unsigned int i = 0; i < child->getNumParents(); ++i)
    {
        testParent = child->getParent(i);
        if (testParent == parent)
            return true;
        else
            return isChild(testParent, parent);
    }
    return false;
}

Interface::DisplayType Interface::getDisplayType()
{
    return _display;
}

Matrix Interface::filterTrackerData(Matrix &h2r)
{
    const float THRESHOLD = 5.0f; // head move speed threshold beyond which tracker values are ignored
    static Matrix prevH2R; // h2r matrix from previous frame
    static bool firstRun = true;
    static double prevTime = cover->currentTime();
    Matrix newH2R; // current h2r matrix
    Vec3 newPos, prevPos; // new and previous head position
    Vec3 diff; // difference in head positions now and previous frame
    double timeNow;
    double ds; // distance head moved since last call
    double dt; // time delta between last and current call

    timeNow = cover->currentTime();

    if (firstRun)
    {
        newH2R = h2r;
        firstRun = false;
    }
    else if (timeNow == prevTime)
        return h2r; // same frame as last one
    else
    {
        newPos = h2r.getTrans();
        prevPos = prevH2R.getTrans();
        diff = newPos - prevPos;
        ds = diff.length();
        dt = timeNow - prevTime;
        if ((ds / dt) < THRESHOLD)
            newH2R = h2r; // check move speed
        else
        {
            newH2R = prevH2R;
            //     cerr << "head tracker data filtered out" << endl;
        }
    }
    prevH2R = newH2R;
    prevTime = timeNow;
    return newH2R;
}

void Interface::setUseZbuffer(int flag)
{
    _zbuf_flag = flag;
}
