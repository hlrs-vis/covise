/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x501 // This specifies WinXP or later - it is needed to access rawmouse from the user32.dll
#endif
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <io.h>
#ifndef PATH_MAX
#define PATH_MAX 512
#endif
#endif

#include <vsg/maths/mat4.h>
#include <vsg/maths/vec3.h>
#include <vsg/utils/ShaderSet.h>
#include <vsg/utils/Builder.h>
#include <vsg/utils/Intersector.h>

#include <list>
#include <ostream>

#include "../../OpenCOVER/OpenVRUI/sginterface/vruiButtons.h"

#include "vvPlugin.h"
#include "units.h"
#include <OpenConfig/access.h>

#include <net/message_types.h>
#include <vsg/nodes/Group.h>
#include <vsg/io/Options.h>
#include <vsg/maths/mat4.h>
#include <array>
#include "vvMathUtils.h"
namespace vive {

namespace ui {
class ButtonGroup;
class Menu;
class Manager;
class VruiView;
}
}

namespace covise {
class MessageBase;
class Message;
class UdpMessage;
}

namespace grmsg {
class coGRMsg;
}

#define MAX_NUMBER_JOYSTICKS 64

namespace vsg
{
class MatrixTransform;
}
namespace vsgText
{
class Font;
}

namespace vrui
{
class coUpdateManager;
class coMenu;
class coToolboxMenu;
class coRowMenu;
}
namespace vrml
{
class Player;
}

namespace covise
{
class TokenBuffer;
}
/// @endcond INTERNAL
namespace vive
{
class vvPlugin;
class vvRenderObject;
class vvInteractor;
class NotifyBuf;
struct Isect
{
    enum IntersectionBits
    {
        Collision = 1,
        Intersection = 2,
        Walk = 4,
        Touch = 8,
        Pick = 16,
        Visible = 32,
        NoMirror = 64,
        Left = 128,
        Right = 256,
        CastShadow = 512,
        ReceiveShadow = 1024,
		Update = 2048,
        vsgEarthSecondary = 0x80000000,
    };

private:
};

namespace Notify
{
enum NotificationLevel
{
    Debug,
    Info,
    Warning,
    Error,
    Fatal
};
}

/*! \class coPointerButton vvPluginSupport.h cover/vvPluginSupport.h
 * Access to buttons and wheel of interaction devices
 */
class VVCORE_EXPORT coPointerButton
{
    friend class vvPluginSupport;
    friend class vvMSController;

public:
    coPointerButton(const std::string &name);
    ~coPointerButton();
    //! button state
    //! @return button press mask
    unsigned int getState() const;
    //! previous button state
    //! @return old button state
    unsigned int oldState() const;
    //! buttons pressed since last frame
    unsigned int wasPressed(unsigned int buttonMask=vrui::vruiButtons::ALL_BUTTONS) const;
    //! buttons released since last frame
    unsigned int wasReleased(unsigned int buttonMask=vrui::vruiButtons::ALL_BUTTONS) const;
    //! is no button pressed
    bool notPressed() const;
    //! accumulated number of wheel events
    int getWheel(size_t idx=0) const;
    //! set number wheel events
    void setWheel(size_t idx, int count);
    //! button name
    const std::string &name() const;

private:
    //! set button state
    void setState(unsigned int);

    unsigned int buttonStatus = 0;
    unsigned int lastStatus = 0;
    int wheelCount[2]={0,0};
    std::string m_name;
};

/*! \class vvPluginSupport vvPluginSupport.h cover/vvPluginSupport.h
 * Provide a stable interface and a single entry point to the most import
 * VIVE functions
 */
class VVCORE_EXPORT vvPluginSupport
{
    friend class VIVE;
    friend class fasi;
    friend class fasi2;
    friend class mopla;
    friend class vvMSController;
    friend class vvIntersection;
    friend class vvFileManager;

public:
    //! returns true if level <= debugLevel
    /*! debug levels should be used like this:
          0 no output,
          1 covise.config entries, vvInit,
          2 constructors, destructors,
          3 all functions which are not called continously,
          4,
          5 all functions which are called continously */
    static vvPluginSupport* instance();
    static void destroy();

    static bool removeChild(vsg::ref_ptr<vsg::Group> &parent, const vsg::ref_ptr<vsg::Node> &child)
    {
        return removeChild(parent.get(), child);
    }
   /* bool removeChild(vsg::Group* parent, const vsg::ref_ptr<vsg::Node>& child)
    {

        return removeChild(parent, child);
    }*/
    static bool removeChild(vsg::Group* parent, const vsg::Node* child)
    {
        if (parent)
        {
            for (auto it = parent->children.begin(); it != parent->children.end(); it++)
            {
                if ((*it).get() == child)
                {
                    parent->children.erase(it);
                    return true;
                }
            }
        }
        return false;
    }
    static bool hasChild(vsg::Group* parent, const vsg::Node* child)
    {
        if (parent)
        {
            for (auto it = parent->children.begin(); it != parent->children.end(); it++)
            {
                if ((*it).get() == child)
                {
                    return true;
                }
            }
        }
        return false;
    }

    vsg::ref_ptr <vsg::ShaderSet> phongShaderSet;
    vsg::ref_ptr<vsg::ShaderSet> getOrCreatePhongShaderSet()
    {
        if (!phongShaderSet)
        {
            phongShaderSet = vsg::createPhongShaderSet(options);
        }
        return phongShaderSet;
    }

    vsg::ref_ptr<vsg::Builder> builder;

    vsg::ref_ptr<vsg::Options> options;

    bool debugLevel(int level) const;
    void initUI();

	void preparePluginUnload();

    //! returns true if clipping is on
    bool isClippingOn() const;

    //! return number of clipping plane user is possibly interacting with
    int getActiveClippingPlane() const;

    //! set number of clipping plane user is possibly interacting with
    void setActiveClippingPlane(int plane);

    // access to scene graph nodes and transformations

    //! get scene group node
    vsg::ref_ptr<vsg::Group> getScene() const;

    //! get the group node for all COVISE and model geometry
    vsg::ref_ptr<vsg::MatrixTransform>  getObjectsRoot() const;

    //! get the MatrixTransform node of the hand
    // (in vvSceneGraph handTransform)
    vsg::MatrixTransform *getPointer() const;

    //! get matrix of hand transform (same as getPointer()->matrix)
    const vsg::dmat4 &getPointerMat() const;
    //       void setPointerMat(vsg::dmat4);

    //! get matrix of current 2D mouse matrix
    //! (the same as getPointerMat for MOUSE tracking)
    const vsg::dmat4 &getMouseMat() const;

    //! get matrix for relative input (identity if no input)
    const vsg::dmat4 &getRelativeMat() const;

    //! get the MatrixTransform for objects translation and rotation
    vsg::MatrixTransform *getObjectsXform() const;

    //! same as getObjectsXform()->matrix
    const vsg::dmat4 &getXformMat() const;

    //! same as getObjectsXform()->matrix = ()
    void setXformMat(const vsg::dmat4 &mat);

    //! get the MatrixTransform for objects scaling
    vsg::MatrixTransform *getObjectsScale() const;

    //! set the scale matrix of the scale node
    void setScale(double s);

    //! get the scale factor of the scale node
    float getScale() const;

    LengthUnit getSceneUnit() const;
    void setSceneUnit(LengthUnit unit);
    void setSceneUnit(const std::string& unitName);
    //! transformation matrix from object coordinates to world coordinates
    /*! multiplied matrices from scene node to objects root node */
    const vsg::dmat4 &getBaseMat() const
    {
        return baseMatrix;
    }

    //! transformation from world coordinates to object coordinates
    /*! use this cached value instead of inverting getBaseMat() yourself */
    const vsg::dmat4 &getInvBaseMat() const;

    //! register filedescriptor fd for watching so that scene will be re-rendererd when it is ready
    void watchFileDescriptor(int fd);
    //! remove fd from filedescriptors to watch
    void unwatchFileDescriptor(int fd);

    vrui::coUpdateManager *getUpdateManager() const;

    //! get the scene size defined in covise.config
    float getSceneSize() const;

    //! favor high-quality rendering instead of interactivity
    bool isHighQuality() const;

    bool isVRBconnected();

    //! send a message either via COVISE connection or via tcp to VRB
    bool sendVrbMessage(const covise::MessageBase *msg) const;

    // tracker data

    //! get the position and orientation of the user in world coordinates
    const vsg::dmat4 &getViewerMat() const;

    //! search geodes under node and set Visible bit in node mask
    void setNodesIsectable(vsg::Node *n, bool isect);

    //! returns a pointer to a coPointerButton object for the main button device
    coPointerButton *getPointerButton() const;

    //! returns a pointer to a coPointerButton object representing the mouse buttons state
    coPointerButton *getMouseButton() const;

    //! returns a pointer to a coPointerButton object representing the buttons state on the relative input device
    coPointerButton *getRelativeButton() const;

    //! returns the COVER Menu (Pinboard)
    vrui::coMenu *getMenu();

    //! return group node of menus
    vsg::Group *getMenuGroup() const;

    // interfacing with plugins

    //! load a new plugin
    vvPlugin *addPlugin(const char *name);

    //! get plugin called name
    vvPlugin *getPlugin(const char *name);

    //! remove the plugin by pointer
    void removePlugin(vvPlugin *);

    //! remove a plugin by name
    int removePlugin(const char *name);

    //! informs other plugins that this plugin extended the scene graph
    void addedNode(vsg::Node *node, vvPlugin *myPlugin);

    //! remove node from the scene graph,
    /*! use this method when removing nodes from the scene graph in order to update
       * VIVE's internal state */
    //! @return if a node was removed
    bool removeNode(vsg::Node *node, bool isGroup = false);

    //! send a message to other plugins
    void sendMessage(vvPlugin *sender, int toWhom, int type, int len, const void *buf);

    //! send a message to a named plugins
    void sendMessage(const vvPlugin *sender, const char *destination, int type, int len, const void *buf, bool localonly = false);

    //! handle coGRMsgs and call guiToRenderMsg method of all plugins
    void guiToRenderMsg(const grmsg::coGRMsg &msg)  const;

    //! grab keyboard input
    /*! other plugins will not get key event notifications,
          returns true if keyboard could be grabbed,
          returns false if keyboard is already grabbed by another plugin */
    bool grabKeyboard(vvPlugin *);

    //! release keyboard input, all plugins will get key events
    void releaseKeyboard(vvPlugin *);

    //! check if keyboard is grabbed
    bool isKeyboardGrabbed();

    //! let plugin request control over viewer position
    bool grabViewer(vvPlugin *);
    //! release control over viewer position
    void releaseViewer(vvPlugin *);
    //! whether a plugins controls viewer position
    bool isViewerGrabbed() const;

    //! forbid saving of scenegraph
    void protectScenegraph();

    //! returns the time in seconds since Jan. 1, 1970 at the beginning of this frame,
    //! use this function if you can since it is faster than currentTime(),
    //! this is the time for which the rendering should be correct,
    //! might differ from system time
    //! @return number of seconds since Jan. 1, 1970 at the beginning of this frame
    double frameTime() const;

    //! returns the duration of the last frame in seconds
    //! @return render time of the last frame in seconds
    double frameDuration() const;

    //! returns the time in seconds since Jan. 1, 1970 at the beginning of this frame,
    //! use this function if you can since it is faster than currentTime(),
    double frameRealTime() const;

    //! returns the current time in seconds since Jan. 1, 1970,
    //! if possible, use frameTime() as it does not require a system call
    //! @return number of seconds since Jan. 1, 1970
    static double currentTime();

    //! get the number of the active cursor shape
    //vsgViewer::GraphicsWindow::MouseCursor getCurrentCursor() const;

    //! set cursor shape
    //! @param type number of cursor shape
    //void setCurrentCursor(vsgViewer::GraphicsWindow::MouseCursor type);

    //! make the cursor visible or invisible
    void setCursorVisible(bool visible) {};

    //! get node currently intersected by pointer
    vsg::Node *getIntersectedNode() const;

    //! get path to node currently intersected by pointer
    const vsg::Intersector::NodePath &getIntersectedNodePath() const;

    //! get world coordinates of intersection hit point
    const vsg::vec3 &getIntersectionHitPointWorld() const;

    //! get normal of intersection hit
    const vsg::vec3 &getIntersectionHitPointWorldNormal() const;

    //! update matrix of an interactor, honouring snapping, ...
    vsg::dmat4 updateInteractorTransform(vsg::dmat4 mat, bool usePointer) const;

    /*********************************************************************/
    // do not use anything beyond this line

    /// @cond INTERNAL
    // deprecated, use coInteraction with high priority instead
    // returns true if another plugin locked the pointer
    int isPointerLocked();

    // old COVISE Messages
    int sendBinMessage(const char *keyword, const char *data, int len);
    // update frametime
    void updateTime();
    // update matrices
    void update();

    //! update internal state related to current person being tracked - called Input system
    void personSwitched(size_t personNumber);

    ui::Manager* ui = nullptr;
    ui::Menu *fileMenu = nullptr;
    ui::Menu *viewOptionsMenu = nullptr;
    ui::Menu *visMenu = nullptr;
    ui::ButtonGroup *navGroup() const;
    ui::VruiView *vruiView = nullptr;

    vsg::dmat4 envCorrectMat;
    vsg::dmat4 invEnvCorrectMat;

    int registerPlayer(vrml::Player *player);
    int unregisterPlayer(vrml::Player *player);
    vrml::Player *usePlayer(void (*playerUnavailableCB)());
    int unusePlayer(void (*playerUnavailableCB)());

    int numJoysticks;
    unsigned char number_buttons[MAX_NUMBER_JOYSTICKS];
    int *buttons[MAX_NUMBER_JOYSTICKS];
    unsigned char number_axes[MAX_NUMBER_JOYSTICKS];
    float *axes[MAX_NUMBER_JOYSTICKS];
    unsigned char number_sliders[MAX_NUMBER_JOYSTICKS];
    float *sliders[MAX_NUMBER_JOYSTICKS];
    unsigned char number_POVs[MAX_NUMBER_JOYSTICKS];
    float *POVs[MAX_NUMBER_JOYSTICKS];

    // utility
    float getSqrDistance(vsg::Node *n, vsg::vec3 &p, vsg::MatrixTransform **path, int pathLength) const;

    vsg::dmat4 *getWorldCoords(vsg::Node *node) const;
    vsg::dvec3 frontScreenCenter; ///< center of front screen
    float frontHorizontalSize; ///< width of front screen
    float frontVerticalSize; ///< height of front screen
    int frontWindowHorizontalSize; ///<  width of front window
    int frontWindowVerticalSize; ///<  width of front window

    //! returns scale factor for interactor-screen distance
    float getInteractorScale(vsg::dvec3 &pos); // pos in World coordinates

    //! returns viewer-screen distance
    float getViewerScreenDistance();

    //restrict interactors to visible scene
    bool restrictOn() const;

    /// @endcond INTERNAL

    /// @cond INTERNAL
    enum MessageDestinations
    {
        TO_ALL,
        TO_ALL_OTHERS,
        TO_SAME,
        TO_SAME_OTHERS,
        VRML_EVENT, // Internal, do not use!!
        NUM_TYPES
    };
    /// @endcond INTERNAL

    vrui::coToolboxMenu *getToolBar(bool create = false);
    void setToolBar(vrui::coToolboxMenu *tb);

    //! use only during vvPlugin::update()
    void setFrameTime(double ft);

    bool sendGrMessage(const grmsg::coGRMsg &grmsg, int msgType = covise::COVISE_MESSAGE_UI) const;

    const config::Access &config() const;
    std::unique_ptr<config::File> configFile(const std::string &path);

private:
    void setFrameRealTime(double ft);

    float scaleFactor = 0; ///< scale depending on viewer-screen FOV
    float viewerDist = 0; ///< distance of viewer from screen
    LengthUnit m_sceneUnit = LengthUnit::Meter; ///< unit in which the scene is specified
    vsg::vec3 eyeToScreen; ///< eye to screen center vector


    mutable vrui::coUpdateManager *updateManager = nullptr;

    mutable int invCalculated;
    vsg::dmat4 handMat;
    bool wasHandValid = false;
    vsg::dmat4 baseMatrix;
    mutable vsg::dmat4 invBaseMatrix;
    double lastFrameStartTime;
    double frameStartTime, frameStartRealTime;
    bool cursorVisible = true;
    vrml::Player *player = nullptr;
    std::set<void (*)()> playerUseList;

    int activeClippingPlane = 0;

    vsg::ref_ptr<vsg::Node> intersectedNode;
    vsg::Intersector::NodePath intersectedNodePath;
    vsg::vec3 intersectionHitPointWorld;
    vsg::vec3 intersectionHitPointWorldNormal;
    vsg::vec3 intersectionHitPointLocal;
    vsg::vec3 intersectionHitPointLocalNormal;
    vsg::dmat4 intersectionMatrix;

    mutable coPointerButton *pointerButton = nullptr;
    mutable coPointerButton *mouseButton = nullptr;
    mutable coPointerButton *relativeButton = nullptr;
    vrui::coToolboxMenu *m_toolBar = nullptr;
    vrui::coMenu *m_vruiMenu = nullptr;
    double interactorScale = 1.;

    int numClipPlanes;

    vvPluginSupport();
    ~vvPluginSupport();

    std::vector<std::ostream *> m_notifyStream;
    std::vector<NotifyBuf *> m_notifyBuf;
    vive::config::Access m_config;
};

VVCORE_EXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &buffer, const vsg::dmat4 &matrix);
VVCORE_EXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &buffer, vsg::dmat4 &matrix);

VVCORE_EXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &buffer, const vsg::vec3 &vec);
VVCORE_EXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &buffer, vsg::vec3 &vec);

//============================================================================
// Useful inline Function Templates
//============================================================================

/// @return the result of value a clamped between left and right
template <class C>
inline C coClamp(const C a, const C left, const C right)
{
    if (a < left)
        return left;
    if (a > right)
        return right;
    return a;
}

extern VVCORE_EXPORT vvPluginSupport *vv;
}
