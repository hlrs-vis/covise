/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvPluginSupport.h"
#include "vvVIVE.h"
#include "vvConfig.h"
#include "vvSceneGraph.h"
#include "vvAnimationManager.h"
#include "vvNavigationManager.h"
#include "vvCollaboration.h"
#include "vvPluginList.h"
#include "vvPlugin.h"
#include "vvMSController.h"
#include "vvFileManager.h"

#include "../OpenConfig/file.h"
#include "../../OpenCOVER/OpenVRUI/coUpdateManager.h"
#include "../../OpenCOVER/OpenVRUI/coInteractionManager.h"
#include "../../OpenCOVER/OpenVRUI/coToolboxMenu.h"
#include <config/CoviseConfig.h>
#include <assert.h>
#include <net/tokenbuffer.h>
#include <net/message.h>
#include <net/udpMessage.h>
#include <net/message_types.h>
#include <../../OpenCOVER/OpenVRUI/sginterface/vruiButtons.h>
#include <../../OpenCOVER/OpenVRUI/vsg/mathUtils.h>
#include <../../OpenCOVER/OpenVRUI/vsg/VSGVruiMatrix.h>
#include <../../OpenCOVER/OpenVRUI/coRowMenu.h>
#include "vvVruiRenderInterface.h"
#include "input/VRKeys.h"
#include "input/input.h"
#include "input/vvMousePointer.h"
#include "vvViewer.h"
#include "ui/Slider.h"
#include <util/unixcompat.h>
#ifdef DOTIMING
#include <util/coTimer.h>
#endif
#include <assert.h>
#ifdef WIN32
#include <windows.h>
#endif
#include <vsg/all.h>

#ifdef __DARWIN_OSX__
#include <Carbon/Carbon.h>
#endif

#include "ui/Menu.h"
#include "ui/Manager.h"
#include "phongShader.cpp"

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

#include <grmsg/coGRMsg.h>
#include <grmsg/coGRKeyWordMsg.h>
#include <grmsg/coGRAnimationOnMsg.h>
#include <grmsg/coGRSetAnimationSpeedMsg.h>
#include <grmsg/coGRSetTimestepMsg.h>
#include <grmsg/coGRSetTrackingParamsMsg.h>
#include <grmsg/coGRPluginMsg.h>
#include <vsgXchange/all.h>


using namespace vrui;
using namespace grmsg;
using namespace covise;

namespace vive
{

vvPluginSupport *vv = NULL;

class NotifyBuf: public std::stringbuf
{
 public:
    NotifyBuf(int level): level(level) {}
    int sync()
    {
        auto s = str();
        if (!s.empty())
        {
           // vvPluginList::instance()->notify(level, s.c_str());
            str("");
        }
        return 0;
    }
 private:
    int level;
};

bool vvPluginSupport::debugLevel(int level) const
{
    return vvConfig::instance()->debugLevel(level);
}

void vvPluginSupport::initUI()
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



vsg::ref_ptr<vsg::MatrixTransform>  vvPluginSupport::getObjectsRoot() const
{
    START("vvPluginSupport::getObjectsRoot");
    return (vvSceneGraph::instance()->objectsRoot());
}

vsg::ref_ptr<vsg::Group> vvPluginSupport::getScene() const
{
    //START("vvPluginSupport::getScene");
    return (vvSceneGraph::instance()->getScene());
}

bool
vvPluginSupport::removeNode(vsg::Node *node, bool isGroup)
{
    (void)isGroup;

    if (!node)
        return false;

   /* if (node->getNumParents() == 0)
        return false;

    vsg::ref_ptr<vsg::Node> n =node;

    while (n->getNumParents() > 0)
    {
        vsg::ref_ptr<vsg::Group> parent = n->getParent(0);
        vsg::ref_ptr<vsg::Node> child = n;

        while (vvSelectionManager::instance()->isHelperNode(parent))
        {
            parent->removeChild(child);
            child = parent;
            parent = parent->getNumParents() > 0 ? parent->getParent(0) : NULL;
        }
        if (parent)
            parent->removeChild(child);
        else
            break;
    }*/
    return true;
}

vsg::Group *
vvPluginSupport::getMenuGroup() const
{
    return (vvSceneGraph::instance()->getMenuGroup());
}

vsg::MatrixTransform *vvPluginSupport::getPointer() const
{
    //START("vvPluginSupport::getPointer");
    return (vvSceneGraph::instance()->getHandTransform());
}

vsg::MatrixTransform *vvPluginSupport::getObjectsXform() const
{
    START("vvPluginSupport::getObjectsXform");
    return (vvSceneGraph::instance()->getTransform());
}

vsg::MatrixTransform *vvPluginSupport::getObjectsScale() const
{
    START("vvPluginSupport::getObjectsScale");
    return (vvSceneGraph::instance()->getScaleTransform());
}

coPointerButton *vvPluginSupport::getMouseButton() const
{
    if (!Input::instance()->hasMouse())
        return NULL;

    if (mouseButton == NULL)
    {
        mouseButton = new coPointerButton("mouse");
    }
    return mouseButton;
    return nullptr;
}

coPointerButton *vvPluginSupport::getRelativeButton() const
{
    if (!Input::instance()->hasRelative())
        return NULL;

    if (relativeButton == NULL)
    {
        relativeButton = new coPointerButton("relative");
    }
    return relativeButton;
    return nullptr;
}

const vsg::dmat4 &vvPluginSupport::getViewerMat() const
{
    START("vvPluginSupport::getViewerMat");
    static vsg::dmat4  transformMatrix;
    return transformMatrix; //return (vvViewer::instance()->getViewerMat());
}

const vsg::dmat4 &vvPluginSupport::getXformMat() const
{
    START("vvPluginSupport::getXformMat");
    return vvSceneGraph::instance()->getTransform()->matrix;
}

const vsg::dmat4 &vvPluginSupport::getMouseMat() const
{
    START("vvPluginSupport::getMouseMat");
    static vsg::dmat4  transformMatrix;
    return transformMatrix; // return Input::instance()->getMouseMat();
}

const vsg::dmat4 &vvPluginSupport::getRelativeMat() const
{

    static vsg::dmat4  transformMatrix;
    return transformMatrix; //return Input::instance()->getRelativeMat();
}

const vsg::dmat4 &vvPluginSupport::getPointerMat() const
{
    //START("vvPluginSupport::getPointerMat");

    if (wasHandValid)
        return handMat;
    else
        return getMouseMat();
}

void vvPluginSupport::setXformMat(const vsg::dmat4 &transformMatrix)
{
    START("vvPluginSupport::setXformMat");
    vvSceneGraph::instance()->getTransform()->matrix = transformMatrix;
    vvCollaboration::instance()->SyncXform();
    vvNavigationManager::instance()->wasJumping();
}

float vvPluginSupport::getSceneSize() const
{
    START("vvPluginSupport::getSceneSize");
    return 1; // return vvConfig::instance()->getSceneSize();
}

// return no. of seconds since epoch 
double vvPluginSupport::currentTime()
{
    START("vvPluginSupport::currentTime");
    struct timeval currentTime;
    gettimeofday(&currentTime, nullptr);

    return currentTime.tv_sec + currentTime.tv_usec / 1000000.0;
}

double vvPluginSupport::frameTime() const
{
    START("vvPluginSupport::frameTime");
    return frameStartTime;
}

double vvPluginSupport::frameDuration() const
{
    return frameStartTime - lastFrameStartTime;
}

double vvPluginSupport::frameRealTime() const
{
    return frameStartRealTime;
}

coMenu *vvPluginSupport::getMenu()
{
    START("vvPluginSupport::getMenu");
    if (!m_vruiMenu)
    {
        if (vv->debugLevel(4))
        {
            fprintf(stderr, "vvPluginSupport::getMenu, creating menu\n");
        }
        vsg::dmat4 dcsTransMat, dcsRotMat, dcsMat, preRot, tmp;
        float xp = 0.0, yp = 0.0, zp = 0.0;
        double h = 0, p = 0, r = 0;
        float size = 1;
        m_vruiMenu = new coRowMenu("COVER");
        m_vruiMenu->setVisible(coCoviseConfig::isOn("VIVE.Menu.Visible", true));

        xp = coCoviseConfig::getFloat("x", "VIVE.Menu.Position", 0.0);
        yp = coCoviseConfig::getFloat("y", "VIVE.Menu.Position", 0.0);
        zp = coCoviseConfig::getFloat("z", "VIVE.Menu.Position", 0.0);
        h = coCoviseConfig::getFloat("h", "VIVE.Menu.Orientation", 0.0);
        p = coCoviseConfig::getFloat("p", "VIVE.Menu.Orientation", 0.0);
        r = coCoviseConfig::getFloat("r", "VIVE.Menu.Orientation", 0.0);
        size = coCoviseConfig::getFloat("VIVE.Menu.Size", 0.0);

        if (size <= 0)
            size = 1;
        dcsRotMat = makeEulerMat(h, p, r);
        preRot = rotate(vsg::radians(90.0f), vsg::vec3(1.0f, 0.0f, 0.0f));
        dcsTransMat= vsg::translate(xp, yp, zp);
        tmp = dcsRotMat* preRot;
        dcsMat = dcsTransMat * tmp;
        VSGVruiMatrix menuMatrix;
        menuMatrix.setMatrix(dcsMat);
        m_vruiMenu->setTransformMatrix(&menuMatrix);
        m_vruiMenu->setScale((float)(size * (getSceneSize() / 2500.0)));
    }

    return m_vruiMenu;
}

void vvPluginSupport::addedNode(vsg::Node *node, vvPlugin *addingPlugin)
{
    START("vvPluginSupport::addedNode");
    vvPluginList::instance()->addNode(node, NULL, addingPlugin);
}

coPointerButton *vvPluginSupport::getPointerButton() const
{
    //START("vvPluginSupport::getPointerButton");
    if (vvConfig::instance()->mouseTracking())
    {
        return getMouseButton();
    }

    if (pointerButton == NULL) // attach Button status as userdata to handTransform
    {
        pointerButton = new coPointerButton("pointer");
    }
    return pointerButton;
}

void vvPluginSupport::setFrameTime(double ft)
{
    frameStartTime = ft;
}

bool vvPluginSupport::sendGrMessage(const coGRMsg &gr, int msgType) const
{
    std::string s = gr.getString();
    Message grmsg{ msgType, DataHandle{const_cast<char *>(s.c_str()), s.length()+1, false} };
    return sendVrbMessage(&grmsg);
}

void vvPluginSupport::setFrameRealTime(double ft)
{
    frameStartRealTime = ft;
}

void vvPluginSupport::updateTime()
{
#ifdef DOTIMING
    MARK0("COVER plugin support update time");
#endif

    lastFrameStartTime = frameStartTime;
    if (vvMSController::instance()->isMaster())
    {
        frameStartRealTime = currentTime();
        if (vvConfig::instance()->constantFrameRate)
        {
            frameStartTime += vvConfig::instance()->constFrameTime;
        }
        else
        {
            frameStartTime = frameStartRealTime;
        }
    }
    
    vvMSController::instance()->syncTime();
#ifdef DOTIMING
    MARK0("done");
#endif
}

void vvPluginSupport::update()
{
    /// START("vvPluginSupport::update");
    if (debugLevel(5))
        fprintf(stderr, "vvPluginSupport::update\n");

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

    auto old = baseMatrix;
    baseMatrix = vvSceneGraph::instance()->getTransform()->matrix * vvSceneGraph::instance()->getScaleTransform()->matrix;

    if (old != baseMatrix && vvMSController::instance()->isMaster())
    {
        coGRKeyWordMsg keyWordMsg("VIEW_CHANGED", false);
        sendGrMessage(keyWordMsg);
    }

#ifdef DOTIMING
    MARK0("done");
#endif

    invCalculated = 0;
    updateManager->update();

    
    if (vvMSController::instance()->isMaster())
    {
        frontScreenCenter = vsg::vec3(0., 0., 0.);
        frontHorizontalSize = 1.;
        frontVerticalSize = 1.;

        frontWindowHorizontalSize = 1024;
        frontWindowVerticalSize = 1024;

        if (vvConfig::instance()->numScreens() > 0)
        {
            // check front screen
            const screenStruct screen = vvConfig::instance()->screens[0];

            frontScreenCenter = screen.xyz;
            frontHorizontalSize = screen.hsize;
            frontVerticalSize = screen.vsize;

            const windowStruct window = vvConfig::instance()->windows[0];
            frontWindowHorizontalSize = window.sx;
            frontWindowVerticalSize = window.sy;
        }

        // get interactor position
        vsg::dvec3 eyePos = getTrans(vv->getViewerMat());

        // save the eye to screen vector
        eyeToScreen = frontScreenCenter - eyePos;

        // calculate distance between interactor and screen
        viewerDist = length(eyeToScreen);

        // use horizontal screen size as normalization factor
        scaleFactor = viewerDist / frontHorizontalSize;
    }

    vvMSController::instance()->syncData((char *)&scaleFactor, sizeof(scaleFactor));
    vvMSController::instance()->syncData((char *)&viewerDist, sizeof(viewerDist));
    vvMSController::instance()->syncData((char *)&eyeToScreen, sizeof(eyeToScreen));

    vvMSController::instance()->syncData((char *)&frontScreenCenter, sizeof(frontScreenCenter));
    vvMSController::instance()->syncData((char *)&frontHorizontalSize, sizeof(frontHorizontalSize));
    vvMSController::instance()->syncData((char *)&frontVerticalSize, sizeof(frontVerticalSize));
}

vvPlugin *vvPluginSupport::addPlugin(const char *name)
{
    START("vvPluginSupport::addPlugin");
    return vvPluginList::instance()->addPlugin(name);
}

vvPlugin *vvPluginSupport::getPlugin(const char *name)
{
    START("vvPluginSupport::getPlugin");
    return vvPluginList::instance()->getPlugin(name);
}

int vvPluginSupport::removePlugin(const char *name)
{
    START("vvPluginSupport::removePlugin");
    vvPlugin *m = vvPluginList::instance()->getPlugin(name);
    if (m)
    {
        vvPluginList::instance()->unload(m);
    }
    return 0;
}

const vsg::dmat4 &vvPluginSupport::getInvBaseMat() const
{
    START("vvPluginSupport::getInvBaseMat");
    if (!invCalculated)
    {
        invBaseMatrix = vsg::inverse(baseMatrix);
        //fprintf(stderr,"vvPluginSupport::getInvBaseMat baseMatrix is singular\n");
        invCalculated = 1;
    }
    return invBaseMatrix;
}

void vvPluginSupport::removePlugin(vvPlugin *m)
{
    START("vvPluginSupport::unload");
    vvPluginList::instance()->unload(m);
}

int vvPluginSupport::isPointerLocked()
{
    //START("vvPluginSupport::isPointerLocked");
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

float vvPluginSupport::getSqrDistance(vsg::Node *n, vsg::vec3 &p,
                                        vsg::MatrixTransform **path = NULL, int pathLength = 0) const
{
   /* START("vvPluginSupport::getSqrDistance");
    vsg::dmat4 mat;
    vsg::BoundingSphere bsphere = n->getBound();
    if (pathLength > 0)
    {
        for (int i = 0; i < pathLength; i++)
        {
            vsg::MatrixTransform *mt = path[i]->asMatrixTransform();
            if (mt)
            {
                mat.postMult(mt->matrix);
            }
        }
    }
    else
    {
        vsg::Node *node = n;
        vsg::Transform *trans = NULL;
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
    vsg::vec3 svec(mat(0, 0), mat(0, 1), mat(0, 2));
    float Scale = svec * svec; // Scale ^2
    vsg::vec3 v;
    if (n->asTransform())
        v = mat.getTrans();
    else
        v = mat.preMult(bsphere.center());
    vsg::vec3 dist = v - p;
    return (dist.length2() / Scale);*/
    return 0;
}

void vvPluginSupport::setScale(double s)
{
    START("vvPluginSupport::setScale");
    vvSceneGraph::instance()->setScaleFactor(s);
    vvNavigationManager::instance()->wasJumping();
}

float vvPluginSupport::getScale() const
{
    START("vvPluginSupport::getScale");
    return vvSceneGraph::instance()->scaleFactor();
}

LengthUnit vvPluginSupport::getSceneUnit() const
{
    return m_sceneUnit;
}

void vvPluginSupport::setSceneUnit(LengthUnit unit)
{
    m_sceneUnit = unit;
}

void vvPluginSupport::setSceneUnit(const std::string& unitName)
{
    auto u = getUnitFromName(unitName);
    if(isValid(u))
        m_sceneUnit = u;
    else
        std::cerr << "warning: " << unitName << " is not a valid length unit" << std::endl;

}

float vvPluginSupport::getInteractorScale(vsg::dvec3 &pos) // pos in World coordinates
{
    
    const vsg::dvec3 eyePos = getTrans(vv->getViewerMat()); 
    const vsg::dvec3 eyeToPos = pos - eyePos;
    double eyeToPosDist = length(eyeToPos);
    double scaleVal;
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

    return (float)(scaleVal / vv->getScale() * interactorScale);
}

float vvPluginSupport::getViewerScreenDistance()
{
    return viewerDist;
}


bool vvPluginSupport::restrictOn() const
{
    return false;// return vvNavigationManager::instance()->restrictOn();
}

coToolboxMenu *vvPluginSupport::getToolBar(bool create)
{
    if (create && !m_toolBar)
    {
        auto tb = new coToolboxMenu("Toolbar");

        //////////////////////////////////////////////////////////
        // position AK-Toolbar and make it visible
        float x = coCoviseConfig::getFloat("x", "VIVE.Plugin.AKToolbar.Position", -400);
        float y = coCoviseConfig::getFloat("y", "VIVE.Plugin.AKToolbar.Position", -200);
        float z = coCoviseConfig::getFloat("z", "VIVE.Plugin.AKToolbar.Position", 0);

        float h = coCoviseConfig::getFloat("h", "VIVE.Plugin.AKToolbar.Orientation", 0);
        float p = coCoviseConfig::getFloat("p", "VIVE.Plugin.AKToolbar.Orientation", 0);
        float r = coCoviseConfig::getFloat("r", "VIVE.Plugin.AKToolbar.Orientation", 0);

        float scale = coCoviseConfig::getFloat("VIVE.Plugin.AKToolbar.Scale", 0.2f);

        int attachment = coUIElement::TOP;
        std::string att = coCoviseConfig::getEntry("VIVE.Plugin.AKToolbar.Attachment");
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

        //float sceneSize = vv->getSceneSize();
        /*
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
        tb->setAttachment(attachment);*/

        m_toolBar = tb;
    }

    return m_toolBar;
}

void vvPluginSupport::setToolBar(coToolboxMenu *tb)
{
    m_toolBar = tb;
}

void vvPluginSupport::preparePluginUnload()
{
	vv->intersectedNode = nullptr;
}
vvPluginSupport *vvPluginSupport::instance()
{
    if (vv == nullptr)
        vv = new vvPluginSupport();
    return vv;
}

void vvPluginSupport::destroy()
{
    delete vv;
    vv = nullptr;
}

vvPluginSupport::vvPluginSupport()
{
    vv = this;

    START("vvPluginSupport::vvPluginSupport");


    new vvVruiRenderInterface();

    ui = new ui::Manager();

    options = vsg::Options::create();
    options->add(vsgXchange::all::create());
    options->shaderSets["phong"] = phongShader();
    auto OptionsFile = vvFileManager::instance()->getName("share/covise/shaders/BuildOptions.vsgt");
    if(OptionsFile)
        options->setValue("read_build_options", OptionsFile);



    char* covisedir = getenv("COVISEDIR");

    options->paths.push_back((std::string(covisedir) + "\\share\\covise").c_str());
    builder = vsg::Builder::create();


    for (int level=0; level<Notify::Fatal; ++level)
    {
        m_notifyBuf.push_back(new NotifyBuf(level));
        m_notifyStream.push_back(new std::ostream(m_notifyBuf[level]));
    }

    /// path for the viewpoint file: initialized by 1st param() call
    intersectedNode = NULL;

    m_toolBar = NULL;
    player = NULL;

    pointerButton = NULL;
    mouseButton = NULL;
    relativeButton = NULL;

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

  //  currentCursor = vsgViewer::GraphicsWindow::LeftArrowCursor;
   // setCurrentCursor(currentCursor);

    frontWindowHorizontalSize = 0;

    if (debugLevel(2))
        fprintf(stderr, "\nnew vvPluginSupport\n");
}

vvPluginSupport::~vvPluginSupport()
{
    START("vvPluginSupport::~vvPluginSupport");
    if (debugLevel(2))
        fprintf(stderr, "delete vvPluginSupport\n");

    if(updateManager)
    updateManager->removeAll();
   // delete vvVruiRenderInterface::the();
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

	intersectedNode = nullptr;
    vv = NULL;

}


void vvPluginSupport::releaseKeyboard(vvPlugin *plugin)
{
    if (vvPluginList::instance()->keyboardGrabber() == plugin)
    {
        vvPluginList::instance()->grabKeyboard(NULL);
    }
}

bool vvPluginSupport::isKeyboardGrabbed()
{
    return (vvPluginList::instance()->keyboardGrabber() != NULL);
}

bool vvPluginSupport::grabKeyboard(vvPlugin *plugin)
{
    if (vvPluginList::instance()->keyboardGrabber() != NULL
        && vvPluginList::instance()->keyboardGrabber() != plugin)
    {
        return false;
    }
    vvPluginList::instance()->grabKeyboard(plugin);
    return true;
}

bool vvPluginSupport::grabViewer(vvPlugin *plugin)
{
    if (vvPluginList::instance()->viewerGrabber()
        && vvPluginList::instance()->viewerGrabber() != plugin)
    {
        return false;
    }
    vvPluginList::instance()->grabViewer(plugin);
    return true;
}

void vvPluginSupport::releaseViewer(vvPlugin *plugin)
{
    if (vvPluginList::instance()->viewerGrabber() == plugin)
    {
        vvPluginList::instance()->grabViewer(NULL);
    }
}

bool vvPluginSupport::isViewerGrabbed() const
{
    return (vvPluginList::instance()->viewerGrabber() != NULL);
}

//-----
void vvPluginSupport::sendMessage(vvPlugin *sender, int toWhom, int type, int len, const void *buf)
{
    START("vvPluginSupport::sendMessage");
    Message message;

    int size = len + 2 * sizeof(int);

    if (toWhom == vvPluginSupport::TO_SAME)
        sender->message(toWhom, type, len, buf);
    if (toWhom == vvPluginSupport::TO_ALL)
        vvPluginList::instance()->message(toWhom, type, len, buf);
    if (toWhom == vvPluginSupport::TO_ALL_OTHERS)
        vvPluginList::instance()->message(toWhom, type, len, buf, sender);

    if ((toWhom == vvPluginSupport::TO_SAME) || (toWhom == vvPluginSupport::TO_SAME_OTHERS))
    {
        size += (int)strlen(sender->getName()) + 1;
        size += 8 - ((strlen(sender->getName()) + 1) % 8);
    }
    message.data = DataHandle(size);
    memcpy(message.data.accessData() + (size - len), buf, len);
    if ((toWhom == vvPluginSupport::TO_SAME) || (toWhom == vvPluginSupport::TO_SAME_OTHERS))
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

    if (!vvMSController::instance()->isSlave())
    {
        vv->sendVrbMessage(&message);
    }
}

void vvPluginSupport::sendMessage(const vvPlugin * /*sender*/, const char *destination, int type, int len, const void *buf, bool localonly)
{
    START("vvPluginSupport::sendMessage");

    //fprintf(stderr,"vvPluginSupport::sendMessage dest=%s\n",destination);

    size_t size = len + 2 * sizeof(int);
    vvPlugin *dest = vvPluginList::instance()->getPlugin(destination);
    if (dest)
    {
        dest->message(0, type, len, buf);
    }
    else if (strcmp(destination, "AKToolbar") != 0)
    {
        cerr << "did not find Plugin " << destination << " in vvPluginSupport::sendMessage" << endl;
    }

    if (!localonly)
    {
        Message message;

        size_t namelen = strlen(destination) + 1;
        namelen += 8 - ((strlen(destination) + 1) % 8);
        size += namelen;
        message.data = DataHandle(size);
        memcpy(message.data.accessData() + (size - len), buf, len);
        memset(message.data.accessData() + 2 * sizeof(int), '\0', namelen);
        strcpy(message.data.accessData() + 2 * sizeof(int), destination);

#ifdef BYTESWAP
        int tmp = vvPluginSupport::TO_SAME;
        byteSwap(tmp);
        ((int *)message.data.accessData())[0] = tmp;
        tmp = type;
        byteSwap(tmp);
        ((int *)message.data.accessData())[1] = tmp;
#else
        ((int *)message->data)[0] = vvPluginSupport::TO_SAME;
        ((int *)message->data)[1] = type;
#endif

        message.type = COVISE_MESSAGE_RENDER_MODULE;

        if (!vvMSController::instance()->isSlave())
        {
            vv->sendVrbMessage(&message);
        }
    }
}
int vvPluginSupport::sendBinMessage(const char *keyword, const char *data, int len)
{
    START("vvPluginSupport::sendBinMessage");
    if (!vvMSController::instance()->isSlave())
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
void vvPluginSupport::guiToRenderMsg(const grmsg::coGRMsg &msg)  const
{
    vvPluginList::instance()->guiToRenderMsg(msg);

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
                vv->addPlugin(pluginMsg.getPlugin());
            }
            else if (act == "unload" || act == "remove")
            {
                vv->removePlugin(pluginMsg.getPlugin());
            }
        }
        break;
        case coGRMsg::ANIMATION_ON:
        {
            auto &animationModeMsg = msg.as<coGRAnimationOnMsg>();
            bool mode = animationModeMsg.getMode() != 0;
            if (vv->debugLevel(3))
                fprintf(stderr, "coGRMsg::ANIMATION_ON mode=%s\n", (mode ? "true" : "false"));
            //vvAnimationManager::instance()->setRemoteAnimate(mode);
        }
        break;
        case coGRMsg::ANIMATION_SPEED:
        {
            auto &animationSpeedMsg = msg.as<coGRSetAnimationSpeedMsg>();
            float speed = animationSpeedMsg.getAnimationSpeed();
            //vvAnimationManager::instance()->setAnimationSpeed(speed);
        }
        break;
        case coGRMsg::ANIMATION_TIMESTEP:
        {
            auto &timestepMsg = msg.as<coGRSetTimestepMsg>();
            int actStep = timestepMsg.getActualTimeStep();
            int maxSteps = timestepMsg.getNumTimeSteps();
            if (vv->debugLevel(3))
                fprintf(stderr, "coGRMsg::ANIMATION_TIMESTEP actStep=%d numSteps=%d\n", actStep, maxSteps);
            if (maxSteps > 0)
            {
             //   vvAnimationManager::instance()->setRemoteAnimationFrame(actStep);
             //   vvAnimationManager::instance()->setNumTimesteps(maxSteps);
            }
        }
        break;
        case coGRMsg::SET_TRACKING_PARAMS:
        {
            auto& trackingMsg = msg.as<coGRSetTrackingParamsMsg>();
            // restrict rotation
            if (trackingMsg.isRotatePoint())
                vvNavigationManager::instance()->setRotationPoint(
                    trackingMsg.getRotatePointX(), trackingMsg.getRotatePointY(), trackingMsg.getRotatePointZ(),
                    trackingMsg.getRotationPointSize());
            else
                vvNavigationManager::instance()->disableRotationPoint();
            if (coCoviseConfig::isOn("VIVE.showRotationPoint", true))
                vvNavigationManager::instance()->setRotationPointVisible(trackingMsg.isRotatePointVisible());
            else
                vvNavigationManager::instance()->setRotationPointVisible(false);
            if (trackingMsg.isRotateAxis())
                vvNavigationManager::instance()->setRotationAxis(
                    trackingMsg.getRotateAxisX(), trackingMsg.getRotateAxisY(), trackingMsg.getRotateAxisZ());
            else
                vvNavigationManager::instance()->disableRotationAxis();
                

            // restrict tranlsation
            if (trackingMsg.isTranslateRestrict())
                vvSceneGraph::instance()->setRestrictBox(trackingMsg.getTranslateMinX(), trackingMsg.getTranslateMaxX(),
                                                        trackingMsg.getTranslateMinY(), trackingMsg.getTranslateMaxY(),
                                                        trackingMsg.getTranslateMinZ(), trackingMsg.getTranslateMaxZ());
            else
                vvSceneGraph::instance()->setRestrictBox(0, 0, 0, 0, 0, 0);
            vvNavigationManager::instance()->setTranslateFactor(trackingMsg.getTranslateFactor());

            // restrict scaling
            if (trackingMsg.isScaleRestrict())
                vvSceneGraph::instance()->setScaleRestrictFactor(trackingMsg.getScaleMin(), trackingMsg.getScaleMax());
            vvSceneGraph::instance()->setScaleFactorButton(trackingMsg.getScaleFactor());
            
            // navigation
            // enable navigationmode showName
            //vvNavigationManager::instance()->setShowName(trackingMsg.isNavModeShowName());
            // set navigationMode
            std::string navmode(trackingMsg.getNavigationMode());
            auto nav = vvNavigationManager::instance();
            nav->setNavMode(navmode);
            //enable tracking in vive

            //vld: VRTracker use. Enable tracking. Add the method in input?
            //vvTracker::instance()->enableTracking(trackingMsg.isTrackingOn());
        }
        break;
        default:
            break;
        }
}

vsg::Node *vvPluginSupport::getIntersectedNode() const
{
    return intersectedNode.get();
}

const vsg::Intersector::NodePath &vvPluginSupport::getIntersectedNodePath() const
{
    return intersectedNodePath;
}

const vsg::vec3 &vvPluginSupport::getIntersectionHitPointWorld() const
{
    return intersectionHitPointWorld;
}

const vsg::vec3 &vvPluginSupport::getIntersectionHitPointWorldNormal() const
{
    return intersectionHitPointWorldNormal;
}
/*
vsg::Matrix vvPluginSupport::updateInteractorTransform(vsg::Matrix mat, bool usePointer) const
{
    if (usePointer && Input::instance()->hasHand() && Input::instance()->isHandValid())
    {
        // get the transformation matrix of the transform
        mat = getPointer()->matrix;
        if (vvNavigationManager::instance()->isSnapping())
        {
            vsg::Matrix w_to_o = vv->getInvBaseMat();
            mat.postMult(w_to_o);
            if (!vvNavigationManager::instance()->isDegreeSnapping())
                snapTo45Degrees(&mat);
            else
                snapToDegrees(vvNavigationManager::instance()->snappingDegrees(), &mat);
            vsg::Matrix o_to_w = vv->getBaseMat();
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
}*/

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

int vvPluginSupport::registerPlayer(vrml::Player *player)
{
    if (this->player)
        return -1;

    this->player = player;
    return 0;
}

int vvPluginSupport::unregisterPlayer(vrml::Player *player)
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

vrml::Player *vvPluginSupport::usePlayer(void (*playerUnavailableCB)())
{
    vv->addPlugin("Vrml97");
    playerUseList.emplace(playerUnavailableCB);
    return this->player;
}

int vvPluginSupport::unusePlayer(void (*playerUnavailableCB)())
{
    auto it = playerUseList.find(playerUnavailableCB);
    if (it == playerUseList.end())
        return -1;

    playerUseList.erase(it);
    return 0;
}

coUpdateManager *vvPluginSupport::getUpdateManager() const
{
    if (!updateManager)
        updateManager = new coUpdateManager();
    return updateManager;
}

int vvPluginSupport::getActiveClippingPlane() const
{
    return activeClippingPlane;
}

void vvPluginSupport::setActiveClippingPlane(int plane)
{
    activeClippingPlane = plane;
}
/*
class IsectVisitor : public vsg::NodeVisitor
{
public:
    IsectVisitor(bool isect)
        : vsg::NodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN)
    {
        isect_ = isect;
    }
    virtual void apply(vsg::Node &node)
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
void vvPluginSupport::setNodesIsectable(vsg::Node *n, bool isect)
{
    IsectVisitor iv(isect);
    if (n)
    {
        n->accept(iv);
    }
}*/
/* see http://www.nps.navy.mil/cs/sullivan/vsgtutorials/vsgGetWorldCoords.htm */

// Visitor to return the world coordinates of a node.
// It traverses from the starting node to the parent.
// The first time it reaches a root node, it stores the world coordinates of
// the node it started from.  The world coordinates are found by concatenating all
// the matrix transforms found on the path from the start node to the root node.
/*
class GetWorldCoordOfNodeVisitor : public vsg::NodeVisitor
{
public:
    GetWorldCoordOfNodeVisitor()
        : vsg::NodeVisitor(NodeVisitor::TRAVERSE_PARENTS)
        , done(false)
    {
        wcMatrix = new vsg::Matrix();
    }
    virtual void apply(vsg::Node &node)
    {
        if (!done)
        {
            if (0 == node.getNumParents()
                         // no parents
                || &node == vv->getObjectsRoot())
            {
                wcMatrix->set(vsg::computeLocalToWorld(this->getNodePath()));
                done = true;
            }
            else
            {
                traverse(node);
            }
        }
    }
    vsg::Matrix *giveUpDaMat()
    {
        return wcMatrix;
    }

private:
    bool done;
    vsg::Matrix *wcMatrix;
};

// Given a valid node placed in a scene under a transform, return the
// world coordinates in an vsg::Matrix.
// Creates a visitor that will update a matrix representing world coordinates
// of the node, return this matrix.
// (This could be a class member for something derived from node also.

vsg::Matrix *vvPluginSupport::getWorldCoords(vsg::Node *node) const
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
*/
bool vvPluginSupport::isHighQuality() const
{
    return vvSceneGraph::instance()->highQuality();
}


bool vvPluginSupport::isVRBconnected()
{
    return false; //return vvMSController::instance()->syncBool(vvVIVE::instance()->isVRBconnected());
}

void vvPluginSupport::protectScenegraph()
{
    vvSceneGraph::instance()->protectScenegraph();
}

bool vvPluginSupport::sendVrbMessage(const covise::MessageBase *msg) const
{
    if(const auto vrbc = vvVIVE::instance()->vrbc())
    {
        vrbc->send(msg);
        return true;
    }

    if (const auto *m = dynamic_cast<const covise::Message *>(msg)) {
        return vvPluginList::instance()->sendVisMessage(m);
    }

    return false;
}

void vvPluginSupport::personSwitched(size_t personNum)
{
    vvViewer::instance()->setSeparation(Input::instance()->eyeDistance());
    vvNavigationManager::instance()->updatePerson();
}

ui::ButtonGroup *vvPluginSupport::navGroup() const
{
    if (vvNavigationManager::instance())
        return vvNavigationManager::instance()->navGroup();

    return nullptr;
}

void vvPluginSupport::watchFileDescriptor(int fd)
{
    vvVIVE::instance()->watchFileDescriptor(fd);
}

void vvPluginSupport::unwatchFileDescriptor(int fd)
{
    vvVIVE::instance()->unwatchFileDescriptor(fd);
}

const config::Access &vvPluginSupport::config() const
{
    return m_config;
}

std::unique_ptr<config::File> vvPluginSupport::configFile(const std::string &path)
{
    return config().file(path);
}

} // namespace vive

covise::TokenBuffer & vive::operator<<(covise::TokenBuffer &buffer, const vsg::dmat4 &matrix)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
    {
        buffer << matrix[r][c];
    }
    return buffer;
}

covise::TokenBuffer &vive::operator>>(covise::TokenBuffer &buffer, vsg::dmat4 &matrix)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
        {
            buffer >>  matrix[r][c];
        }
    return buffer;
}

covise::TokenBuffer & vive::operator<<(covise::TokenBuffer &buffer, const vsg::vec3 &vec)
{
    for (int ctr = 0; ctr < 3; ++ctr)
    {
        buffer << vec[ctr];
    }
    return buffer;
}

covise::TokenBuffer & vive::operator>>(covise::TokenBuffer &buffer, vsg::vec3 &vec)
{
    for (int ctr = 0; ctr < 3; ++ctr)
    {
        buffer >> vec[ctr];
    }
    return buffer;
}
