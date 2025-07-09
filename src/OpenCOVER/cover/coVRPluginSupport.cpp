/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRPluginSupport.h"
#include "OpenCOVER.h"
#include "coVRSelectionManager.h"
#include "VRSceneGraph.h"
#include "coVRAnimationManager.h"
#include "coVRNavigationManager.h"
#include "coVRCollaboration.h"
#include "coVRPluginList.h"
#include "coVRPlugin.h"
#include "coVRMSController.h"

#include <OpenConfig/file.h>
#include <OpenVRUI/coUpdateManager.h>
#include <OpenVRUI/coInteractionManager.h>
#include <OpenVRUI/coToolboxMenu.h>
#include <config/CoviseConfig.h>
#include <assert.h>
#include <net/tokenbuffer.h>
#include <net/message.h>
#include <net/udpMessage.h>
#include <net/message_types.h>
#include <OpenVRUI/sginterface/vruiButtons.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>
#include <OpenVRUI/coRowMenu.h>
#include "VRVruiRenderInterface.h"
#include "input/VRKeys.h"
#include "input/input.h"
#include "input/coMousePointer.h"
#include "VRViewer.h"
#include "ui/Slider.h"
#ifdef DOTIMING
#include <util/coTimer.h>
#endif

#include <osg/MatrixTransform>
#include <osg/BoundingSphere>
#include <osg/Material>
#include <osg/ClipNode>
#include <osg/PolygonMode>
#include <util/unixcompat.h>
#include <osg/ComputeBoundsVisitor>

#ifdef __DARWIN_OSX__
#include <Carbon/Carbon.h>
#endif

#include "ui/Menu.h"
#include "ui/Manager.h"

// undef this to get no START/END messaged
#undef VERBOSE

#ifdef VERBOSE
#define START(msg) fprintf(stderr, "START: %s\n", (msg))
#define END(msg) fprintf(stderr, "END:   %s\n", (msg))
#else
#define START(msg)
#define END(msg)
#endif

#include <vrb/client/VRBClient.h>
#include "coVRConfig.h"

#include <grmsg/coGRMsg.h>
#include <grmsg/coGRKeyWordMsg.h>
#include <grmsg/coGRAnimationOnMsg.h>
#include <grmsg/coGRSetAnimationSpeedMsg.h>
#include <grmsg/coGRSetTimestepMsg.h>
#include <grmsg/coGRSetTrackingParamsMsg.h>
#include <grmsg/coGRPluginMsg.h>


using namespace vrui;
using namespace grmsg;
using namespace covise;

namespace opencover
{

coVRPluginSupport *cover = NULL;

class NotifyBuf: public std::stringbuf
{
 public:
    NotifyBuf(int level): level(level) {}
    int sync()
    {
        auto s = str();
        if (!s.empty())
        {
            coVRPluginList::instance()->notify(level, s.c_str());
            str("");
        }
        return 0;
    }
 private:
    int level;
};

bool coVRPluginSupport::debugLevel(int level) const
{
    return coVRConfig::instance()->debugLevel(level);
}

void coVRPluginSupport::initUI()
{
    fileMenu = new ui::Menu("File", ui);
    viewOptionsMenu = new ui::Menu("ViewOptions", ui);
    viewOptionsMenu->setText("View options");
    viewOptionsMenu->allowRelayout(true);

    auto interactorScaleSlider = new ui::Slider(viewOptionsMenu, "InteractorScale");
    interactorScaleSlider->setText("Interactor scale");
    interactorScaleSlider->setVisible(false, ui::View::VR);
    interactorScaleSlider->setBounds(0.01, 100.);
    interactorScaleSlider->setValue(1.);
    interactorScaleSlider->setScale(ui::Slider::Logarithmic);
    interactorScaleSlider->setCallback([this](double value, bool released) {
        interactorScale = value;
    });

    ui->init();
}

std::ostream &coVRPluginSupport::notify(Notify::NotificationLevel level) const
{
    std::ostream *str = m_notifyStream[0];
    if (level < m_notifyStream.size())
        str = m_notifyStream[level];
    *str << std::flush;

    return *str;
}

std::ostream &coVRPluginSupport::notify(Notify::NotificationLevel level, const char *fmt, ...) const
{
    std::vector<char> text(strlen(fmt)+500);

    va_list args;
    va_start(args, fmt);
    int messageSize = vsnprintf(&text[0], text.size(), fmt, args);
	va_end(args);
	if (messageSize>text.size())
	{
        text.resize(strlen(fmt)+messageSize);
		va_start(args, fmt);
		vsnprintf(&text[0], text.size(), fmt, args);
		va_end(args);
	}

    return notify(level) << &text[0];
}

osg::ClipNode *coVRPluginSupport::getObjectsRoot() const
{
    START("coVRPluginSupport::getObjectsRoot");
    return (VRSceneGraph::instance()->objectsRoot());
    return NULL;
}

osg::Group *coVRPluginSupport::getScene() const
{
    //START("coVRPluginSupport::getScene");
    return (VRSceneGraph::instance()->getScene());
}

bool
coVRPluginSupport::removeNode(osg::Node *node, bool isGroup)
{
    (void)isGroup;

    if (!node)
        return false;

    if (node->getNumParents() == 0)
        return false;

    osg::ref_ptr<osg::Node> n =node;

    while (n->getNumParents() > 0)
    {
        osg::ref_ptr<osg::Group> parent = n->getParent(0);
        osg::ref_ptr<osg::Node> child = n;

        while (coVRSelectionManager::instance()->isHelperNode(parent))
        {
            parent->removeChild(child);
            child = parent;
            parent = parent->getNumParents() > 0 ? parent->getParent(0) : NULL;
        }
        if (parent)
            parent->removeChild(child);
        else
            break;
    }
    return true;
}

osg::Group *
coVRPluginSupport::getMenuGroup() const
{
    return (VRSceneGraph::instance()->getMenuGroup());
}

osg::MatrixTransform *coVRPluginSupport::getPointer() const
{
    //START("coVRPluginSupport::getPointer");
    return (VRSceneGraph::instance()->getHandTransform());
}

osg::MatrixTransform *coVRPluginSupport::getObjectsXform() const
{
    START("coVRPluginSupport::getObjectsXform");
    return (VRSceneGraph::instance()->getTransform());
}

osg::MatrixTransform *coVRPluginSupport::getObjectsScale() const
{
    START("coVRPluginSupport::getObjectsScale");
    return (VRSceneGraph::instance()->getScaleTransform());
}

coPointerButton *coVRPluginSupport::getMouseButton() const
{
    if (!Input::instance()->hasMouse())
        return NULL;

    if (mouseButton == NULL)
    {
        mouseButton = new coPointerButton("mouse");
    }
    return mouseButton;
}

coPointerButton *coVRPluginSupport::getRelativeButton() const
{
    if (!Input::instance()->hasRelative())
        return NULL;

    if (relativeButton == NULL)
    {
        relativeButton = new coPointerButton("relative");
    }
    return relativeButton;
}

const osg::Matrix& coVRPluginSupport::getViewerMat(int i) const
{
    START("coVRPluginSupport::getViewerMat");
    return (VRViewer::instance()->getViewerMat(i));
}

const osg::Matrix &coVRPluginSupport::getXformMat() const
{
    START("coVRPluginSupport::getXformMat");
    static osg::Matrix transformMatrix;
    transformMatrix = VRSceneGraph::instance()->getTransform()->getMatrix();
    return transformMatrix;
}

const osg::Matrix &coVRPluginSupport::getMouseMat() const
{
    START("coVRPluginSupport::getMouseMat");
    return Input::instance()->getMouseMat();
}

const osg::Matrix &coVRPluginSupport::getRelativeMat() const
{
    return Input::instance()->getRelativeMat();
}

const osg::Matrix &coVRPluginSupport::getPointerMat() const
{
    //START("coVRPluginSupport::getPointerMat");

    if (wasHandValid)
        return handMat;
    else
        return getMouseMat();
}

void coVRPluginSupport::setXformMat(const osg::Matrix &transformMatrix)
{
    START("coVRPluginSupport::setXformMat");
    VRSceneGraph::instance()->getTransform()->setMatrix(transformMatrix);
    coVRCollaboration::instance()->SyncXform();
    coVRNavigationManager::instance()->wasJumping();
}

float coVRPluginSupport::getSceneSize() const
{
    START("coVRPluginSupport::getSceneSize");
    return coVRConfig::instance()->getSceneSize();
}

// return no. of seconds since epoch 
double coVRPluginSupport::currentTime()
{
    START("coVRPluginSupport::currentTime");
    struct timeval currentTime;
    gettimeofday(&currentTime, nullptr);

    return currentTime.tv_sec + currentTime.tv_usec / 1000000.0;
}

double coVRPluginSupport::frameTime() const
{
    START("coVRPluginSupport::frameTime");
    return frameStartTime;
}

double coVRPluginSupport::frameDuration() const
{
    return frameStartTime - lastFrameStartTime;
}

double coVRPluginSupport::frameRealTime() const
{
    return frameStartRealTime;
}

coMenu *coVRPluginSupport::getMenu()
{
    START("coVRPluginSupport::getMenu");
    if (!m_vruiMenu)
    {
        if (cover->debugLevel(4))
        {
            fprintf(stderr, "coVRPluginSupport::getMenu, creating menu\n");
        }
        osg::Matrix dcsTransMat, dcsRotMat, dcsMat, preRot, tmp;
        float xp = 0.0, yp = 0.0, zp = 0.0;
        float h = 0, p = 0, r = 0;
        float size = 1;
        m_vruiMenu = new coRowMenu("COVER");
        m_vruiMenu->setVisible(coCoviseConfig::isOn("COVER.Menu.Visible", true));

        xp = coCoviseConfig::getFloat("x", "COVER.Menu.Position", 0.0);
        yp = coCoviseConfig::getFloat("y", "COVER.Menu.Position", 0.0);
        zp = coCoviseConfig::getFloat("z", "COVER.Menu.Position", 0.0);
        h = coCoviseConfig::getFloat("h", "COVER.Menu.Orientation", 0.0);
        p = coCoviseConfig::getFloat("p", "COVER.Menu.Orientation", 0.0);
        r = coCoviseConfig::getFloat("r", "COVER.Menu.Orientation", 0.0);
        size = coCoviseConfig::getFloat("COVER.Menu.Size", 0.0);

        if (size <= 0)
            size = 1;
        dcsRotMat.makeIdentity();
        MAKE_EULER_MAT(dcsRotMat, h, p, r);
        preRot.makeRotate(osg::inDegrees(90.0f), 1.0f, 0.0f, 0.0f);
        dcsTransMat.makeTranslate(xp, yp, zp);
        tmp.mult(preRot, dcsRotMat);
        dcsMat.mult(tmp, dcsTransMat);
        OSGVruiMatrix menuMatrix;
        menuMatrix.setMatrix(dcsMat);
        m_vruiMenu->setTransformMatrix(&menuMatrix);
        m_vruiMenu->setScale(size * (getSceneSize() / 2500.0));
    }

    return m_vruiMenu;
}

void coVRPluginSupport::addedNode(osg::Node *node, coVRPlugin *addingPlugin)
{
    START("coVRPluginSupport::addedNode");
    coVRPluginList::instance()->addNode(node, NULL, addingPlugin);
}

coPointerButton *coVRPluginSupport::getPointerButton() const
{
    //START("coVRPluginSupport::getPointerButton");
    if (coVRConfig::instance()->mouseTracking())
    {
        return getMouseButton();
    }

    if (pointerButton == NULL) // attach Button status as userdata to handTransform
    {
        pointerButton = new coPointerButton("pointer");
    }
    return pointerButton;
}

void coVRPluginSupport::setFrameTime(double ft)
{
    frameStartTime = ft;
}

void coVRPluginSupport::setRenderStrategy(osg::Drawable *draw, bool dynamic)
{
    bool displaylist = coVRConfig::instance()->useDisplayLists() && !dynamic;
    bool vbo = !displaylist && coVRConfig::instance()->useVBOs();
    //bool vao = false;

    draw->setUseDisplayList(displaylist);
    draw->setUseVertexBufferObjects(vbo);
    //draw->setUseVertexArrayObject(vao);
}

bool coVRPluginSupport::sendGrMessage(const coGRMsg &gr, int msgType) const
{
    std::string s = gr.getString();
    Message grmsg{ msgType, DataHandle{const_cast<char *>(s.c_str()), s.length()+1, false} };
    return sendVrbMessage(&grmsg);
}

void coVRPluginSupport::setFrameRealTime(double ft)
{
    frameStartRealTime = ft;
}

void coVRPluginSupport::updateTime()
{
#ifdef DOTIMING
    MARK0("COVER plugin support update time");
#endif

    lastFrameStartTime = frameStartTime;
    if (coVRMSController::instance()->isMaster())
    {
        frameStartRealTime = currentTime();
        if (coVRConfig::instance()->constantFrameRate)
        {
            frameStartTime += coVRConfig::instance()->constFrameTime;
        }
        else
        {
            frameStartTime = frameStartRealTime;
        }
    }
    
    coVRMSController::instance()->syncTime();
#ifdef DOTIMING
    MARK0("done");
#endif
}

void coVRPluginSupport::update()
{
    /// START("coVRPluginSupport::update");
    if (debugLevel(5))
        fprintf(stderr, "coVRPluginSupport::update\n");

    for (auto nb: m_notifyBuf) {
        if (nb)
            nb->sync();
    }

    if (Input::instance()->hasHand() && Input::instance()->isHandValid())
    {
        wasHandValid = true;
        handMat = Input::instance()->getHandMat();
    }

    if (getRelativeButton())
    {
        getRelativeButton()->setState(Input::instance()->getRelativeButtonState());
    }

    if (getPointerButton() && getPointerButton()!=getMouseButton())
    {
        for (size_t i=0; i<2; ++i)
            getPointerButton()->setWheel(i, 0);
        getPointerButton()->setState(Input::instance()->getButtonState());
#if 0
        if (getPointerButton()->wasPressed() || getPointerButton()->wasReleased() || getPointerButton()->getState())
            std::cerr << "pointer pressed: " << getPointerButton()->wasPressed() << ", released: " << getPointerButton()->wasReleased() << ", state: " << getPointerButton()->getState() << std::endl;
#endif
    }

    if (getMouseButton())
    {
        for (size_t i=0; i<2; ++i)
            getMouseButton()->setWheel(i, Input::instance()->mouse()->wheel(i));
        getMouseButton()->setState(Input::instance()->mouse()->buttonState());
#if 0
        if (getMouseButton()->wasPressed() || getMouseButton()->wasReleased() || getMouseButton()->getState())
            std::cerr << "mouse pressed: " << getMouseButton()->wasPressed() << ", released: " << getMouseButton()->wasReleased() << ", state: " << getMouseButton()->getState() << std::endl;
#endif

        size_t currentPerson = Input::instance()->getActivePerson();
        if ((getMouseButton()->wasPressed(vruiButtons::PERSON_NEXT))
                || (getPointerButton()->wasPressed(vruiButtons::PERSON_NEXT)))
        {
            ++currentPerson;
            currentPerson %= Input::instance()->getNumPersons();
            Input::instance()->setActivePerson(currentPerson);
        }
        if ((getMouseButton()->wasPressed(vruiButtons::PERSON_PREV))
                || (getPointerButton()->wasPressed(vruiButtons::PERSON_PREV)))
        {
            if (currentPerson == 0)
                currentPerson = Input::instance()->getNumPersons();
            --currentPerson;
            Input::instance()->setActivePerson(currentPerson);
        }
    }

#ifdef DOTIMING
    MARK0("COVER update matrices and button status in plugin support class");
#endif

    osg::Matrix old = baseMatrix;
    baseMatrix = VRSceneGraph::instance()->getScaleTransform()->getMatrix();
    osg::Matrix transformMatrix = VRSceneGraph::instance()->getTransform()->getMatrix();
    baseMatrix.postMult(transformMatrix);

    if (old != baseMatrix && coVRMSController::instance()->isMaster())
    {
        coGRKeyWordMsg keyWordMsg("VIEW_CHANGED", false);
        sendGrMessage(keyWordMsg);
    }

#ifdef DOTIMING
    MARK0("done");
#endif

    invCalculated = 0;
    updateManager->update();
    //get rotational part of Xform only
    osg::Matrix frontRot(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);
    if (VRViewer::instance()->isMatrixOverwriteOn())
    {
        envCorrectMat.makeIdentity();
    }
    else if (coVRConfig::instance()->m_envMapMode == coVRConfig::FIXED_TO_VIEWER)
    {
        osg::Quat rot;
        rot.set(getXformMat());
        osg::Matrix rotMat;
        rotMat.makeRotate(rot);
        envCorrectMat = rotMat;
    }
    else if (coVRConfig::instance()->m_envMapMode == coVRConfig::FIXED_TO_OBJROOT)
    {
        envCorrectMat = getXformMat();
    }
    else if (coVRConfig::instance()->m_envMapMode == coVRConfig::FIXED_TO_VIEWER_FRONT)
    {
        osg::Quat rot;
        rot.set(getXformMat());
        osg::Matrix rotMat;
        rotMat.makeRotate(rot);
        envCorrectMat = frontRot * rotMat;
    }
    else if (coVRConfig::instance()->m_envMapMode == coVRConfig::FIXED_TO_OBJROOT_FRONT)
    {
        envCorrectMat = frontRot * getXformMat();
    }
    else if (coVRConfig::instance()->m_envMapMode == coVRConfig::NONE)
    {
        envCorrectMat = frontRot * getXformMat();
    }
    invEnvCorrectMat.invert(envCorrectMat);
    //draw.baseMat = getBaseMat();
    //draw.invBaseMat = getInvBaseMat();

    //START("coVRPluginSupport::getInteractorScale");
    if (coVRMSController::instance()->isMaster())
    {
        frontScreenCenter = osg::Vec3(0., 0., 0.);
        frontHorizontalSize = 1.;
        frontVerticalSize = 1.;

        frontWindowHorizontalSize = 1024;
        frontWindowVerticalSize = 1024;

        if (coVRConfig::instance()->numScreens() > 0)
        {
            // check front screen
            const screenStruct screen = coVRConfig::instance()->screens[0];

            frontScreenCenter = screen.xyz;
            frontHorizontalSize = screen.hsize;
            frontVerticalSize = screen.vsize;

            const windowStruct window = coVRConfig::instance()->windows[0];
            frontWindowHorizontalSize = window.sx;
            frontWindowVerticalSize = window.sy;
        }

        // get interactor position
        const osg::Vec3d eyePos = cover->getViewerMat().getTrans();

        // save the eye to screen vector
        eyeToScreen = frontScreenCenter - eyePos;

        // calculate distance between interactor and screen
        viewerDist = eyeToScreen.length();

        // use horizontal screen size as normalization factor
        scaleFactor = viewerDist / frontHorizontalSize;
    }

    coVRMSController::instance()->syncData((char *)&scaleFactor, sizeof(scaleFactor));
    coVRMSController::instance()->syncData((char *)&viewerDist, sizeof(viewerDist));
    coVRMSController::instance()->syncData((char *)&eyeToScreen, sizeof(eyeToScreen));

    coVRMSController::instance()->syncData((char *)&frontScreenCenter, sizeof(frontScreenCenter));
    coVRMSController::instance()->syncData((char *)&frontHorizontalSize, sizeof(frontHorizontalSize));
    coVRMSController::instance()->syncData((char *)&frontVerticalSize, sizeof(frontVerticalSize));
}

coVRPlugin *coVRPluginSupport::addPlugin(const char *name)
{
    START("coVRPluginSupport::addPlugin");
    return coVRPluginList::instance()->addPlugin(name);
}

coVRPlugin *coVRPluginSupport::getPlugin(const char *name)
{
    START("coVRPluginSupport::getPlugin");
    return coVRPluginList::instance()->getPlugin(name);
}

int coVRPluginSupport::removePlugin(const char *name)
{
    START("coVRPluginSupport::removePlugin");
    coVRPlugin *m = coVRPluginList::instance()->getPlugin(name);
    if (m)
    {
        coVRPluginList::instance()->unload(m);
    }
    return 0;
}

const osg::Matrix &coVRPluginSupport::getInvBaseMat() const
{
    START("coVRPluginSupport::getInvBaseMat");
    if (!invCalculated)
    {
        invBaseMatrix.invert(baseMatrix);
        //fprintf(stderr,"coVRPluginSupport::getInvBaseMat baseMatrix is singular\n");
        invCalculated = 1;
    }
    return invBaseMatrix;
}

void coVRPluginSupport::removePlugin(coVRPlugin *m)
{
    START("coVRPluginSupport::unload");
    coVRPluginList::instance()->unload(m);
}

int coVRPluginSupport::isPointerLocked()
{
    //START("coVRPluginSupport::isPointerLocked");
    bool isLocked;
    isLocked = coInteractionManager::the()->isOneActive(coInteraction::ButtonA);
    if (isLocked)
        return true;
    isLocked = coInteractionManager::the()->isOneActive(coInteraction::ButtonB);
    if (isLocked)
        return true;
    isLocked = coInteractionManager::the()->isOneActive(coInteraction::ButtonC);
    if (isLocked)
        return true;
    return (false);
}

float coVRPluginSupport::getSqrDistance(osg::Node *n, osg::Vec3 &p,
                                        osg::MatrixTransform **path = NULL, int pathLength = 0) const
{
    START("coVRPluginSupport::getSqrDistance");
    osg::Matrix mat;
    mat.makeIdentity();
    osg::BoundingSphere bsphere = n->getBound();
    if (pathLength > 0)
    {
        for (int i = 0; i < pathLength; i++)
        {
            osg::MatrixTransform *mt = path[i]->asMatrixTransform();
            if (mt)
            {
                mat.postMult(mt->getMatrix());
            }
        }
    }
    else
    {
        osg::Node *node = n;
        osg::Transform *trans = NULL;
        while (node)
        {
            if ((trans = node->asTransform()))
                break;
            node = node->getParent(0);
        }
        if (trans)
        {
            trans->asTransform()->computeLocalToWorldMatrix(mat, NULL);
        }
    }
    osg::Vec3 svec(mat(0, 0), mat(0, 1), mat(0, 2));
    float Scale = svec * svec; // Scale ^2
    osg::Vec3 v;
    if (n->asTransform())
        v = mat.getTrans();
    else
        v = mat.preMult(bsphere.center());
    osg::Vec3 dist = v - p;
    return (dist.length2() / Scale);
}

void coVRPluginSupport::setScale(float s)
{
    START("coVRPluginSupport::setScale");
    VRSceneGraph::instance()->setScaleFactor(s);
    coVRNavigationManager::instance()->wasJumping();
}

float coVRPluginSupport::getScale() const
{
    START("coVRPluginSupport::getScale");
    return VRSceneGraph::instance()->scaleFactor();
}

LengthUnit coVRPluginSupport::getSceneUnit() const
{
    return m_sceneUnit;
}

void coVRPluginSupport::setSceneUnit(LengthUnit unit)
{
    m_sceneUnit = unit;
}

void coVRPluginSupport::setSceneUnit(const std::string& unitName)
{
    auto u = getUnitFromName(unitName);
    if(isValid(u))
        m_sceneUnit = u;
    else
        std::cerr << "warning: " << unitName << " is not a valid length unit" << std::endl;

}

float coVRPluginSupport::getInteractorScale(osg::Vec3 &pos) // pos in World coordinates
{
    const osg::Vec3 eyePos = cover->getViewerMat().getTrans();
    const osg::Vec3 eyeToPos = pos - eyePos;
    float eyeToPosDist = eyeToPos.length();
    float scaleVal;
    if (eyeToPosDist < 2000)
    {
        scaleVal = 1.0;
    }
    else
    {
        //float oeffnungswinkel = frontHorizontalSize/(frontScreenCenter[1]-eyePos[1]);
        //scaleVal = ((eyeToPosDist) * oeffnungswinkel/(oeffnungswinkel*2000));

        scaleVal = eyeToPosDist / (2000);
    }

    return scaleVal / cover->getScale() * interactorScale;
}

float coVRPluginSupport::getViewerScreenDistance()
{
    return viewerDist;
}

osg::BoundingBox coVRPluginSupport::getBBox(osg::Node *node) const
{
    osg::ComputeBoundsVisitor cbv;
    cbv.setTraversalMask(Isect::Visible);
    node->accept(cbv);
    const osg::NodePath path = cbv.getNodePath();
    //fprintf(stderr, "!!!! nodepath %d\n", (int)path.size());
    //for (osg::NodePath::const_iterator it = path.begin(); it != path.end(); it++)
    //fprintf(stderr, "    node - %s(%s)\n", (*it)->getName(), (*it)->className());
    return cbv.getBoundingBox();
}

bool coVRPluginSupport::restrictOn() const
{
    return coVRNavigationManager::instance()->restrictOn();
}

coToolboxMenu *coVRPluginSupport::getToolBar(bool create)
{
    if (create && !m_toolBar)
    {
        auto tb = new coToolboxMenu("Toolbar");

        //////////////////////////////////////////////////////////
        // position AK-Toolbar and make it visible
        float x = coCoviseConfig::getFloat("x", "COVER.Plugin.AKToolbar.Position", -400);
        float y = coCoviseConfig::getFloat("y", "COVER.Plugin.AKToolbar.Position", -200);
        float z = coCoviseConfig::getFloat("z", "COVER.Plugin.AKToolbar.Position", 0);

        float h = coCoviseConfig::getFloat("h", "COVER.Plugin.AKToolbar.Orientation", 0);
        float p = coCoviseConfig::getFloat("p", "COVER.Plugin.AKToolbar.Orientation", 0);
        float r = coCoviseConfig::getFloat("r", "COVER.Plugin.AKToolbar.Orientation", 0);

        float scale = coCoviseConfig::getFloat("COVER.Plugin.AKToolbar.Scale", 0.2);

        int attachment = coUIElement::TOP;
        std::string att = coCoviseConfig::getEntry("COVER.Plugin.AKToolbar.Attachment");
        if (att != "")
        {
            if (!strcasecmp(att.c_str(), "BOTTOM"))
            {
                attachment = coUIElement::BOTTOM;
            }
            else if (!strcasecmp(att.c_str(), "LEFT"))
            {
                attachment = coUIElement::LEFT;
            }
            else if (!strcasecmp(att.c_str(), "RIGHT"))
            {
                attachment = coUIElement::RIGHT;
            }
        }

        //float sceneSize = cover->getSceneSize();

        vruiMatrix *mat = vruiRendererInterface::the()->createMatrix();
        vruiMatrix *rot = vruiRendererInterface::the()->createMatrix();
        vruiMatrix *trans = vruiRendererInterface::the()->createMatrix();

        rot->makeEuler(h, p, r);
        trans->makeTranslate(x, y, z);
        mat->makeIdentity();
        mat->mult(rot);
        mat->mult(trans);
        tb->setTransformMatrix(mat);
        tb->setScale(scale);
        tb->setVisible(true);
        tb->fixPos(true);
        tb->setAttachment(attachment);

        m_toolBar = tb;
    }

    return m_toolBar;
}

void coVRPluginSupport::setToolBar(coToolboxMenu *tb)
{
    m_toolBar = tb;
}

void coVRPluginSupport::preparePluginUnload()
{
	cover->intersectedDrawable = nullptr; // intersectedDrawable might be a node from this plugin
	cover->intersectedNode = nullptr;
}

coVRPluginSupport::coVRPluginSupport()
{
    assert(!cover);
    cover = this;

    START("coVRPluginSupport::coVRPluginSupport");

    new VRVruiRenderInterface();

    ui = new ui::Manager();


    for (int level=0; level<Notify::Fatal; ++level)
    {
        m_notifyBuf.push_back(new NotifyBuf(level));
        m_notifyStream.push_back(new std::ostream(m_notifyBuf[level]));
    }

    /// path for the viewpoint file: initialized by 1st param() call
    intersectedNode = NULL;

    m_toolBar = NULL;
    numClipPlanes = coCoviseConfig::getInt("COVER.NumClipPlanes", 3);
    for (int i = 0; i < numClipPlanes; i++)
    {
        clipPlanes[i] = new osg::ClipPlane();
        clipPlanes[i]->setClipPlaneNum(i);
    }
    NoFrameBuffer = new osg::ColorMask(false, false, false, false);
    player = NULL;

    pointerButton = NULL;
    mouseButton = NULL;
    relativeButton = NULL;
    baseMatrix.makeIdentity();

    invCalculated = false;
    frameStartTime = 0.0;
    frameStartRealTime = 0.0;

    // Init Joystick data
    numJoysticks = 0;
    for (int i = 0; i < MAX_NUMBER_JOYSTICKS; i++)
    {
        number_buttons[i] = 0;
        number_sliders[i] = 0;
        number_axes[i] = 0;
        number_POVs[i] = 0;
        buttons[i] = NULL;
        sliders[i] = NULL;
        axes[i] = NULL;
        POVs[i] = NULL;
    }

    currentCursor = osgViewer::GraphicsWindow::LeftArrowCursor;
    setCurrentCursor(currentCursor);

    frontWindowHorizontalSize = 0;

    if (debugLevel(2))
        fprintf(stderr, "\nnew coVRPluginSupport\n");
}

coVRPluginSupport::~coVRPluginSupport()
{
    START("coVRPluginSupport::~coVRPluginSupport");
    if (debugLevel(2))
        fprintf(stderr, "delete coVRPluginSupport\n");

    updateManager->removeAll();
    delete VRVruiRenderInterface::the();
    delete updateManager;

    while(!m_notifyStream.empty())
    {
        delete m_notifyStream.back();
        m_notifyStream.pop_back();
    }
    while(!m_notifyBuf.empty())
    {
        delete m_notifyBuf.back();
        m_notifyBuf.pop_back();
    }

	intersectedDrawable = nullptr;
	intersectedNode = nullptr;
    cover = NULL;

}

int coVRPluginSupport::getNumClipPlanes()
{
    return numClipPlanes;
}

void coVRPluginSupport::releaseKeyboard(coVRPlugin *plugin)
{
    if (coVRPluginList::instance()->keyboardGrabber() == plugin)
    {
        coVRPluginList::instance()->grabKeyboard(NULL);
    }
}

bool coVRPluginSupport::isKeyboardGrabbed()
{
    return (coVRPluginList::instance()->keyboardGrabber() != NULL);
}

bool coVRPluginSupport::grabKeyboard(coVRPlugin *plugin)
{
    if (coVRPluginList::instance()->keyboardGrabber() != NULL
        && coVRPluginList::instance()->keyboardGrabber() != plugin)
    {
        return false;
    }
    coVRPluginList::instance()->grabKeyboard(plugin);
    return true;
}

bool coVRPluginSupport::grabViewer(coVRPlugin *plugin)
{
    if (coVRPluginList::instance()->viewerGrabber()
        && coVRPluginList::instance()->viewerGrabber() != plugin)
    {
        return false;
    }
    coVRPluginList::instance()->grabViewer(plugin);
    return true;
}

void coVRPluginSupport::releaseViewer(coVRPlugin *plugin)
{
    if (coVRPluginList::instance()->viewerGrabber() == plugin)
    {
        coVRPluginList::instance()->grabViewer(NULL);
    }
}

bool coVRPluginSupport::isViewerGrabbed() const
{
    return (coVRPluginList::instance()->viewerGrabber() != NULL);
}

// get the active cursor number
osgViewer::GraphicsWindow::MouseCursor coVRPluginSupport::getCurrentCursor() const
{
    return currentCursor;
}

// set the active cursor number
void coVRPluginSupport::setCurrentCursor(osgViewer::GraphicsWindow::MouseCursor cursor)
{
    if (currentCursor == cursor)
        return;
    currentCursor = cursor;
    for (int i = 0; i < coVRConfig::instance()->numWindows(); i++)
    {
        if (coVRConfig::instance()->windows[i].window && cursorVisible)
            coVRConfig::instance()->windows[i].window->setCursor(currentCursor);
    }
}

// make the cursor visible or invisible
void coVRPluginSupport::setCursorVisible(bool visible)
{
    if (visible != cursorVisible)
    {
        for (int i = 0; i < coVRConfig::instance()->numWindows(); i++)
        {
            if (coVRConfig::instance()->windows[i].window)
            {
                if (visible)
                    coVRConfig::instance()->windows[i].window->setCursor(currentCursor);
                coVRConfig::instance()->windows[i].window->useCursor(visible);
            }
        }
    }
    cursorVisible = visible;
}

//-----
void coVRPluginSupport::sendMessage(coVRPlugin *sender, int toWhom, int type, int len, const void *buf)
{
    START("coVRPluginSupport::sendMessage");
    Message message;

    int size = len + 2 * sizeof(int);

    if (toWhom == coVRPluginSupport::TO_SAME)
        sender->message(toWhom, type, len, buf);
    if (toWhom == coVRPluginSupport::TO_ALL)
        coVRPluginList::instance()->message(toWhom, type, len, buf);
    if (toWhom == coVRPluginSupport::TO_ALL_OTHERS)
        coVRPluginList::instance()->message(toWhom, type, len, buf, sender);

    if ((toWhom == coVRPluginSupport::TO_SAME) || (toWhom == coVRPluginSupport::TO_SAME_OTHERS))
    {
        size += strlen(sender->getName()) + 1;
        size += 8 - ((strlen(sender->getName()) + 1) % 8);
    }
    message.data = DataHandle(size);
    memcpy(message.data.accessData() + (size - len), buf, len);
    if ((toWhom == coVRPluginSupport::TO_SAME) || (toWhom == coVRPluginSupport::TO_SAME_OTHERS))
    {
        strcpy((message.data.accessData() + 2 * sizeof(int)), sender->getName());
    }
#ifdef BYTESWAP
    int tmp = toWhom;
    byteSwap(tmp);
    ((int *)message.data.accessData())[0] = tmp;
    tmp = type;
    byteSwap(tmp);
    ((int *)message.data.accessData())[1] = tmp;
#else
    ((int *)message->data)[0] = toWhom;
    ((int *)message->data)[1] = type;
#endif

    message.type = COVISE_MESSAGE_RENDER_MODULE;

    if (!coVRMSController::instance()->isSlave())
    {
        cover->sendVrbMessage(&message);
    }
}

void coVRPluginSupport::sendMessage(const coVRPlugin * /*sender*/, const char *destination, int type, int len, const void *buf, bool localonly)
{
    START("coVRPluginSupport::sendMessage");

    //fprintf(stderr,"coVRPluginSupport::sendMessage dest=%s\n",destination);

    int size = len + 2 * sizeof(int);
    coVRPlugin *dest = coVRPluginList::instance()->getPlugin(destination);
    if (dest)
    {
        dest->message(0, type, len, buf);
    }
    else if (strcmp(destination, "AKToolbar") != 0)
    {
        cerr << "did not find Plugin " << destination << " in coVRPluginSupport::sendMessage" << endl;
    }

    if (!localonly)
    {
        Message message;

        int namelen = strlen(destination) + 1;
        namelen += 8 - ((strlen(destination) + 1) % 8);
        size += namelen;
        message.data = DataHandle(size);
        memcpy(message.data.accessData() + (size - len), buf, len);
        memset(message.data.accessData() + 2 * sizeof(int), '\0', namelen);
        strcpy(message.data.accessData() + 2 * sizeof(int), destination);

#ifdef BYTESWAP
        int tmp = coVRPluginSupport::TO_SAME;
        byteSwap(tmp);
        ((int *)message.data.accessData())[0] = tmp;
        tmp = type;
        byteSwap(tmp);
        ((int *)message.data.accessData())[1] = tmp;
#else
        ((int *)message->data)[0] = coVRPluginSupport::TO_SAME;
        ((int *)message->data)[1] = type;
#endif

        message.type = COVISE_MESSAGE_RENDER_MODULE;

        if (!coVRMSController::instance()->isSlave())
        {
            cover->sendVrbMessage(&message);
        }
    }
}
int coVRPluginSupport::sendBinMessage(const char *keyword, const char *data, int len)
{
    START("coVRPluginSupport::sendBinMessage");
    if (!coVRMSController::instance()->isSlave())
    {
        size_t size = strlen(keyword) + 2;
        size += len;

        Message message{ Message::RENDER, DataHandle{size} };
        message.data.accessData()[0] = 0;
        strcpy(&message.data.accessData()[1], keyword);
        memcpy(&message.data.accessData()[strlen(keyword) + 2], data, len);

        bool ret = sendVrbMessage(&message);
        return ret ? 1 : 0;
    }

    return 1;
}

//! handle coGRMsgs and call guiToRenderMsg method of all plugins
void coVRPluginSupport::guiToRenderMsg(const grmsg::coGRMsg &msg)  const
{
    coVRPluginList::instance()->guiToRenderMsg(msg);

    if (!msg.isValid())
        return;

    switch(msg.getType())
    {
        case coGRMsg::PLUGIN:
        {
            auto &pluginMsg = msg.as<coGRPluginMsg>();
            std::string act(pluginMsg.getAction());
            if (act == "load" || act == "add")
            {
                cover->addPlugin(pluginMsg.getPlugin());
            }
            else if (act == "unload" || act == "remove")
            {
                cover->removePlugin(pluginMsg.getPlugin());
            }
        }
        break;
        case coGRMsg::ANIMATION_ON:
        {
            auto &animationModeMsg = msg.as<coGRAnimationOnMsg>();
            bool mode = animationModeMsg.getMode() != 0;
            if (cover->debugLevel(3))
                fprintf(stderr, "coGRMsg::ANIMATION_ON mode=%s\n", (mode ? "true" : "false"));
            coVRAnimationManager::instance()->setRemoteAnimate(mode);
        }
        break;
        case coGRMsg::ANIMATION_SPEED:
        {
            auto &animationSpeedMsg = msg.as<coGRSetAnimationSpeedMsg>();
            float speed = animationSpeedMsg.getAnimationSpeed();
            coVRAnimationManager::instance()->setAnimationSpeed(speed);
        }
        break;
        case coGRMsg::ANIMATION_TIMESTEP:
        {
            auto &timestepMsg = msg.as<coGRSetTimestepMsg>();
            int actStep = timestepMsg.getActualTimeStep();
            int maxSteps = timestepMsg.getNumTimeSteps();
            if (cover->debugLevel(3))
                fprintf(stderr, "coGRMsg::ANIMATION_TIMESTEP actStep=%d numSteps=%d\n", actStep, maxSteps);
            if (maxSteps > 0)
            {
                coVRAnimationManager::instance()->setRemoteAnimationFrame(actStep);
                coVRAnimationManager::instance()->setNumTimesteps(maxSteps);
            }
        }
        break;
        case coGRMsg::SET_TRACKING_PARAMS:
        {
            auto &trackingMsg = msg.as<coGRSetTrackingParamsMsg>();
            // restrict rotation
            if (trackingMsg.isRotatePoint())
                coVRNavigationManager::instance()->setRotationPoint(
                    trackingMsg.getRotatePointX(), trackingMsg.getRotatePointY(), trackingMsg.getRotatePointZ(),
                    trackingMsg.getRotationPointSize());
            else
                coVRNavigationManager::instance()->disableRotationPoint();
            if (coCoviseConfig::isOn("COVER.showRotationPoint", true))
                coVRNavigationManager::instance()->setRotationPointVisible(trackingMsg.isRotatePointVisible());
            else
                coVRNavigationManager::instance()->setRotationPointVisible(false);
            if (trackingMsg.isRotateAxis())
                coVRNavigationManager::instance()->setRotationAxis(
                    trackingMsg.getRotateAxisX(), trackingMsg.getRotateAxisY(), trackingMsg.getRotateAxisZ());
            else
                coVRNavigationManager::instance()->disableRotationAxis();

            // restrict tranlsation
            if (trackingMsg.isTranslateRestrict())
                VRSceneGraph::instance()->setRestrictBox(trackingMsg.getTranslateMinX(), trackingMsg.getTranslateMaxX(),
                                                        trackingMsg.getTranslateMinY(), trackingMsg.getTranslateMaxY(),
                                                        trackingMsg.getTranslateMinZ(), trackingMsg.getTranslateMaxZ());
            else
                VRSceneGraph::instance()->setRestrictBox(0, 0, 0, 0, 0, 0);
            coVRNavigationManager::instance()->setTranslateFactor(trackingMsg.getTranslateFactor());

            // restrict scaling
            if (trackingMsg.isScaleRestrict())
                VRSceneGraph::instance()->setScaleRestrictFactor(trackingMsg.getScaleMin(), trackingMsg.getScaleMax());
            VRSceneGraph::instance()->setScaleFactorButton(trackingMsg.getScaleFactor());

            // navigation
            // enable navigationmode showName
            //coVRNavigationManager::instance()->setShowName(trackingMsg.isNavModeShowName());
            // set navigationMode
            std::string navmode(trackingMsg.getNavigationMode());
            auto nav = coVRNavigationManager::instance();
            nav->setNavMode(navmode);
            //enable tracking in opencover

            //vld: VRTracker use. Enable tracking. Add the method in input?
            //VRTracker::instance()->enableTracking(trackingMsg.isTrackingOn());
        }
        break;
        default:
            break;
        }
}

osg::Node *coVRPluginSupport::getIntersectedNode() const
{
    return intersectedNode.get();
}

const osg::NodePath &coVRPluginSupport::getIntersectedNodePath() const
{
    return intersectedNodePath;
}

const osg::Vec3 &coVRPluginSupport::getIntersectionHitPointWorld() const
{
    return intersectionHitPointWorld;
}

const osg::Vec3 &coVRPluginSupport::getIntersectionHitPointWorldNormal() const
{
    return intersectionHitPointWorldNormal;
}

osg::Matrix coVRPluginSupport::updateInteractorTransform(osg::Matrix mat, bool usePointer) const
{
    if (usePointer && Input::instance()->hasHand() && Input::instance()->isHandValid())
    {
        // get the transformation matrix of the transform
        mat = getPointer()->getMatrix();
        if (coVRNavigationManager::instance()->isSnapping())
        {
            osg::Matrix w_to_o = cover->getInvBaseMat();
            mat.postMult(w_to_o);
            if (!coVRNavigationManager::instance()->isDegreeSnapping())
                snapTo45Degrees(&mat);
            else
                snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &mat);
            osg::Matrix o_to_w = cover->getBaseMat();
            mat.postMult(o_to_w);
            coCoord coord = mat;
            coord.makeMat(mat);
        }
    }

    if (Input::instance()->hasRelative() && Input::instance()->isRelativeValid())
    {
        auto rel = Input::instance()->getRelativeMat();
        auto tr = rel.getTrans();
        coCoord coord(rel);
        MAKE_EULER_MAT_VEC(rel, -coord.hpr);
        rel.setTrans(-tr);
        mat *= rel;
    }

    return mat;
}

coPointerButton::coPointerButton(const std::string &name)
    : m_name(name)
{
    START("coPointerButton::coPointerButton");
    buttonStatus = 0;
    lastStatus = 0;
    wheelCount[0] = wheelCount[1] = 0;
}

coPointerButton::~coPointerButton()
{
    START("coPointerButton::~coPointerButton");
}

const std::string &coPointerButton::name() const
{

    return m_name;
}

unsigned int coPointerButton::getState() const
{
    //START("coPointerButton::getButtonStatus")
    return buttonStatus;
}

unsigned int coPointerButton::oldState() const
{
    START("coPointerButton::oldButtonStatus");
    return lastStatus;
}

bool coPointerButton::notPressed() const
{
    START("coPointerButton::notPressed");
    return (buttonStatus == 0);
}

unsigned int coPointerButton::wasPressed(unsigned int buttonMask) const
{
    return buttonMask & ((getState() ^ oldState()) & getState());
}

unsigned int coPointerButton::wasReleased(unsigned int buttonMask) const
{
    return buttonMask & ((getState() ^ oldState()) & oldState());
}

void coPointerButton::setState(unsigned int newButton) // called from
{
    //START("coPointerButton::setButtonStatus");
    lastStatus = buttonStatus;
    buttonStatus = newButton;
}

int coPointerButton::getWheel(size_t idx) const
{
    if (idx >= 2)
        return 0;
    return wheelCount[idx];
}

void coPointerButton::setWheel(size_t idx, int count)
{
    if (idx >= 2)
        return;
    wheelCount[idx] = count;
}

int coVRPluginSupport::registerPlayer(vrml::Player *player)
{
    if (this->player)
        return -1;

    this->player = player;
    return 0;
}

int coVRPluginSupport::unregisterPlayer(vrml::Player *player)
{
    if (this->player != player)
        return -1;

    for (auto cb: playerUseList)
    {
        if (cb)
            cb();
    }

    player = NULL;

    return 0;
}

vrml::Player *coVRPluginSupport::usePlayer(void (*playerUnavailableCB)())
{
    cover->addPlugin("Vrml97");
    playerUseList.emplace(playerUnavailableCB);
    return this->player;
}

int coVRPluginSupport::unusePlayer(void (*playerUnavailableCB)())
{
    auto it = playerUseList.find(playerUnavailableCB);
    if (it == playerUseList.end())
        return -1;

    playerUseList.erase(it);
    return 0;
}

coUpdateManager *coVRPluginSupport::getUpdateManager() const
{
    if (!updateManager)
        updateManager = new coUpdateManager();
    return updateManager;
}

bool coVRPluginSupport::isClippingOn() const
{
    return getObjectsRoot()->getNumClipPlanes() > 0;
}

int coVRPluginSupport::getActiveClippingPlane() const
{
    return activeClippingPlane;
}

void coVRPluginSupport::setActiveClippingPlane(int plane)
{
    activeClippingPlane = plane;
}

class IsectVisitor : public osg::NodeVisitor
{
public:
    IsectVisitor(bool isect)
        : osg::NodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN)
    {
        isect_ = isect;
    }
    virtual void apply(osg::Node &node)
    {
        if (isect_)
            node.setNodeMask(node.getNodeMask() | (Isect::Intersection));
        else
            node.setNodeMask(node.getNodeMask() & (~Isect::Intersection));
        traverse(node);
    }

private:
    bool isect_;
};
void coVRPluginSupport::setNodesIsectable(osg::Node *n, bool isect)
{
    IsectVisitor iv(isect);
    if (n)
    {
        n->accept(iv);
    }
}
/* see http://www.nps.navy.mil/cs/sullivan/osgtutorials/osgGetWorldCoords.htm */

// Visitor to return the world coordinates of a node.
// It traverses from the starting node to the parent.
// The first time it reaches a root node, it stores the world coordinates of
// the node it started from.  The world coordinates are found by concatenating all
// the matrix transforms found on the path from the start node to the root node.

class GetWorldCoordOfNodeVisitor : public osg::NodeVisitor
{
public:
    GetWorldCoordOfNodeVisitor()
        : osg::NodeVisitor(NodeVisitor::TRAVERSE_PARENTS)
        , done(false)
    {
        wcMatrix = new osg::Matrix();
    }
    virtual void apply(osg::Node &node)
    {
        if (!done)
        {
            if (0 == node.getNumParents()
                         // no parents
                || &node == cover->getObjectsRoot())
            {
                wcMatrix->set(osg::computeLocalToWorld(this->getNodePath()));
                done = true;
            }
            else
            {
                traverse(node);
            }
        }
    }
    osg::Matrix *giveUpDaMat()
    {
        return wcMatrix;
    }

private:
    bool done;
    osg::Matrix *wcMatrix;
};

// Given a valid node placed in a scene under a transform, return the
// world coordinates in an osg::Matrix.
// Creates a visitor that will update a matrix representing world coordinates
// of the node, return this matrix.
// (This could be a class member for something derived from node also.

osg::Matrix *coVRPluginSupport::getWorldCoords(osg::Node *node) const
{
    GetWorldCoordOfNodeVisitor ncv;
    if (node)
    {
        node->accept(ncv);
        return ncv.giveUpDaMat();
    }
    else
    {
        return NULL;
    }
}

bool coVRPluginSupport::isHighQuality() const
{
    return VRSceneGraph::instance()->highQuality();
}


bool coVRPluginSupport::isVRBconnected()
{
    return coVRMSController::instance()->syncBool(OpenCOVER::instance()->isVRBconnected());
}

void coVRPluginSupport::protectScenegraph()
{
    VRSceneGraph::instance()->protectScenegraph();
}

bool coVRPluginSupport::sendVrbMessage(const covise::MessageBase *msg) const
{
    if(const auto vrbc = OpenCOVER::instance()->vrbc())
    {
        vrbc->send(msg);
        return true;
    }

    if (const auto *m = dynamic_cast<const covise::Message *>(msg)) {
        return coVRPluginList::instance()->sendVisMessage(m);
    }

    return false;
}

void coVRPluginSupport::personSwitched(size_t personNum)
{
    VRViewer::instance()->setSeparation(Input::instance()->eyeDistance());
    coVRNavigationManager::instance()->updatePerson();
}

ui::ButtonGroup *coVRPluginSupport::navGroup() const
{
    if (coVRNavigationManager::instance())
        return coVRNavigationManager::instance()->navGroup();

    return nullptr;
}

void coVRPluginSupport::watchFileDescriptor(int fd)
{
    OpenCOVER::instance()->watchFileDescriptor(fd);
}

void coVRPluginSupport::unwatchFileDescriptor(int fd)
{
    OpenCOVER::instance()->unwatchFileDescriptor(fd);
}

const config::Access &coVRPluginSupport::config() const
{
    return m_config;
}

std::unique_ptr<config::File> coVRPluginSupport::configFile(const std::string &path)
{
    return config().file(path);
}

} // namespace opencover

covise::TokenBuffer &opencover::operator<<(covise::TokenBuffer &buffer, const osg::Matrixd &matrix)
{
    for (int ctr = 0; ctr < 16; ++ctr)
    {
        buffer << matrix.ptr()[ctr];
    }
    return buffer;
}

covise::TokenBuffer &opencover::operator>>(covise::TokenBuffer &buffer, osg::Matrixd &matrix)
{
    double array[16];
    for (int ctr = 0; ctr < 16; ++ctr)
    {
        buffer >> array[ctr];
    }
    matrix.set(array);
    return buffer;
}

covise::TokenBuffer &opencover::operator<<(covise::TokenBuffer &buffer, const osg::Vec3f &vec)
{
    for (int ctr = 0; ctr < 3; ++ctr)
    {
        buffer << vec[ctr];
    }
    return buffer;
}

covise::TokenBuffer &opencover::operator>>(covise::TokenBuffer &buffer, osg::Vec3f &vec)
{
    for (int ctr = 0; ctr < 3; ++ctr)
    {
        buffer >> vec[ctr];
    }
    return buffer;
}
