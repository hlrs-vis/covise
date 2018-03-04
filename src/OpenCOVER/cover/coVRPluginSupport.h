/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPEN_COVER_PLUGIN_SUPPORT
#define OPEN_COVER_PLUGIN_SUPPORT

/*! \file
 \brief  provide a stable interface to the most important OpenCOVER classes and calls
         through a single pointer

 \author
 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

/// @cond INTERNAL

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

#include <limits.h>
#include <osg/Matrix>
#include <osg/Geode>
#include <osg/ClipNode>
#include <osgViewer/GraphicsWindow>
#include <osg/BoundingBox>

#include <deque>
#include <list>
#include <ostream>
#include <OpenVRUI/sginterface/vruiButtons.h>
#include "coVRPlugin.h"

#include "ui/Manager.h"
#include "ui/Menu.h"
#include "ui/ButtonGroup.h"
#include "ui/VruiView.h"

namespace opencover {
namespace ui {
class Menu;
}
}

#define MAX_NUMBER_JOYSTICKS 64

namespace osg
{
class MatrixTransform;
}

namespace osgText
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
namespace opencover
{
class coVRPlugin;
class RenderObject;
class coInteractor;
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
        OsgEarthSecondary = 0x80000000,
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

/*! \class coPointerButton coVRPluginSupport.h cover/coVRPluginSupport.h
 * Access to buttons and wheel of interaction devices
 */
class COVEREXPORT coPointerButton
{
    friend class coVRPluginSupport;
    friend class coVRMSController;

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

/*! \class coVRPluginSupport coVRPluginSupport.h cover/coVRPluginSupport.h
 * Provide a stable interface and a single entry point to the most import
 * OpenCOVER functions
 */
class COVEREXPORT coVRPluginSupport
{
    friend class OpenCOVER;
    friend class coVRMSController;
    friend class coIntersection;

public:
    //! returns true if level <= debugLevel
    /*! debug levels should be used like this:
          0 no output,
          1 covise.config entries, coVRInit,
          2 constructors, destructors,
          3 all functions which are not called continously,
          4,
          5 all functions which are called continously */
    bool debugLevel(int level) const;

    // show a message to the user
    std::ostream &notify(Notify::NotificationLevel level=Notify::Info) const;
    std::ostream &notify(Notify::NotificationLevel level, const char *format, ...) const
#ifdef __GNUC__
        __attribute__((format(printf, 3, 4)))
#endif
        ;

    // OpenGL clipping
    /// @cond INTERNAL
    enum
    {
        MAX_NUM_CLIP_PLANES = 6
    };
    /// @endcond INTERNAL

    //! return the number of clipPlanes reserved for the kernel, others are available to VRML ClippingPlane Node
    int getNumClipPlanes();

    //! return pointer to a clipping plane
    osg::ClipPlane *getClipPlane(int num)
    {
        return clipPlanes[num].get();
    }

    //! returns true if clipping is on
    bool isClippingOn() const;

    //! return number of clipping plane user is possibly interacting with
    int getActiveClippingPlane() const;

    //! set number of clipping plane user is possibly interacting with
    void setActiveClippingPlane(int plane);

    // access to scene graph nodes and transformations

    //! get scene group node
    osg::Group *getScene() const;

    //! get the group node for all COVISE and model geometry
    osg::ClipNode *getObjectsRoot() const;

    //! get the MatrixTransform node of the hand
    // (in VRSceneGraph handTransform)
    osg::MatrixTransform *getPointer() const;

    //! get matrix of hand transform (same as getPointer()->getMatrix())
    const osg::Matrix &getPointerMat() const;
    //       void setPointerMat(osg::Matrix);

    //! get matrix of current 2D mouse matrix
    //! (the same as getPointerMat for MOUSE tracking)
    const osg::Matrix &getMouseMat() const;

    //! get matrix for relative input (identity if no input)
    const osg::Matrix &getRelativeMat() const;

    //! get the MatrixTransform for objects translation and rotation
    osg::MatrixTransform *getObjectsXform() const;

    //! same as getObjectsXform()->getMatrix()
    const osg::Matrix &getXformMat() const;

    //! same as getObjectsXform()->setMatrix()
    void setXformMat(const osg::Matrix &mat);

    //! get the MatrixTransform for objects scaling
    osg::MatrixTransform *getObjectsScale() const;

    //! set the scale matrix of the scale node
    void setScale(float s);

    //! get the scale factor of the scale node
    float getScale() const;

    //! transformation matrix from object coordinates to world coordinates
    /*! multiplied matrices from scene node to objects root node */
    const osg::Matrix &getBaseMat() const
    {
        return baseMatrix;
    }

    //! transformation from world coordinates to object coordinates
    /*! use this cached value instead of inverting getBaseMat() yourself */
    const osg::Matrix &getInvBaseMat() const;

    vrui::coUpdateManager *getUpdateManager() const;

    //! get the scene size defined in covise.config
    float getSceneSize() const;

    //! favor high-quality rendering instead of interactivity
    bool isHighQuality() const;

    bool isVRBconnected();

    //! send a message either via COVISE connection or via VRB
    bool sendVrbMessage(const covise::Message *msg) const;

    // tracker data

    //! get the position and orientation of the user in world coordinates
    const osg::Matrix &getViewerMat() const;

    //! search geodes under node and set Visible bit in node mask
    void setNodesIsectable(osg::Node *n, bool isect);

    //! returns a pointer to a coPointerButton object for the main button device
    coPointerButton *getPointerButton() const;

    //! returns a pointer to a coPointerButton object representing the mouse buttons state
    coPointerButton *getMouseButton() const;

    //! returns a pointer to a coPointerButton object representing the buttons state on the relative input device
    coPointerButton *getRelativeButton() const;

    //! returns the COVER Menu (Pinboard)
    vrui::coMenu *getMenu();

    //! return group node of menus
    osg::Group *getMenuGroup() const;

    // interfacing with plugins

    //! load a new plugin
    coVRPlugin *addPlugin(const char *name);

    //! get plugin called name
    coVRPlugin *getPlugin(const char *name);

    //! remove the plugin by pointer
    void removePlugin(coVRPlugin *);

    //! remove a plugin by name
    int removePlugin(const char *name);

    //! informs other plugins that this plugin extended the scene graph
    void addedNode(osg::Node *node, coVRPlugin *myPlugin);

    //! remove node from the scene graph,
    /*! use this method when removing nodes from the scene graph in order to update
       * OpenCOVER's internal state */
    //! @return if a node was removed
    bool removeNode(osg::Node *node, bool isGroup = false);

    //! send a message to other plugins
    void sendMessage(coVRPlugin *sender, int toWhom, int type, int len, const void *buf);

    //! send a message to a named plugins
    void sendMessage(coVRPlugin *sender, const char *destination, int type, int len, const void *buf, bool localonly = false);
    //! grab keyboard input
    /*! other plugins will not get key event notifications,
          returns true if keyboard could be grabbed,
          returns false if keyboard is already grabbed by another plugin */
    bool grabKeyboard(coVRPlugin *);

    //! release keyboard input, all plugins will get key events
    void releaseKeyboard(coVRPlugin *);

    //! check if keyboard is grabbed
    bool isKeyboardGrabbed();

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
    osgViewer::GraphicsWindow::MouseCursor getCurrentCursor() const;

    //! set cursor shape
    //! @param type number of cursor shape
    void setCurrentCursor(osgViewer::GraphicsWindow::MouseCursor type);

    //! make the cursor visible or invisible
    void setCursorVisible(bool visible);

    //! get node currently intersected by pointer
    osg::Node *getIntersectedNode() const;

    //! get path to node currently intersected by pointer
    const osg::NodePath &getIntersectedNodePath() const;

    //! get world coordinates of intersection hit point
    const osg::Vec3 &getIntersectionHitPointWorld() const;

    //! get normal of intersection hit
    const osg::Vec3 &getIntersectionHitPointWorldNormal() const;

    //! update matrix of an interactor, honouring snapping, ...
    osg::Matrix updateInteractorTransform(osg::Matrix mat, bool usePointer) const;

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

    ui::Manager *ui = nullptr;
    ui::Menu *fileMenu = nullptr;
    ui::Menu *viewOptionsMenu = nullptr;
    ui::Menu *visMenu = nullptr;
    ui::ButtonGroup *navGroup() const;
    ui::VruiView *vruiView = nullptr;

    osg::Matrix envCorrectMat;
    osg::Matrix invEnvCorrectMat;

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

    osg::ref_ptr<osg::ColorMask> getNoFrameBuffer()
    {
        return NoFrameBuffer;
    }

    // utility
    float getSqrDistance(osg::Node *n, osg::Vec3 &p, osg::MatrixTransform **path, int pathLength) const;

    osg::Matrix *getWorldCoords(osg::Node *node) const;
    osg::Vec3 frontScreenCenter; ///< center of front screen
    float frontHorizontalSize; ///< width of front screen
    float frontVerticalSize; ///< height of front screen
    int frontWindowHorizontalSize; ///<  width of front window
    int frontWindowVerticalSize; ///<  width of front window

    //! returns scale factor for interactor-screen distance
    float getInteractorScale(osg::Vec3 &pos); // pos in World coordinates

    //! returns viewer-screen distance
    float getViewerScreenDistance();

    //! compute the box of all visible nodes above and included node
    osg::BoundingBox getBBox(osg::Node *node) const;

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

    //! use only during coVRPlugin::update()
    void setFrameTime(double ft);

private:
    void setFrameRealTime(double ft);

    //! calls the callback
    void callButtonCallback(const char *buttonName);

    float scaleFactor; ///< scale depending on viewer-screen FOV
    float viewerDist; ///< distance of viewer from screen
    osg::Vec3 eyeToScreen; ///< eye to screen center vector

    osg::ref_ptr<osg::ColorMask> NoFrameBuffer;

    osg::ref_ptr<osg::ClipPlane> clipPlanes[MAX_NUM_CLIP_PLANES];

    mutable vrui::coUpdateManager *updateManager;

    mutable int invCalculated;
    osg::Matrix handMat;
    bool wasHandValid = false;
    osg::Matrix baseMatrix;
    mutable osg::Matrix invBaseMatrix;
    double lastFrameStartTime;
    double frameStartTime, frameStartRealTime;
    osgViewer::GraphicsWindow::MouseCursor currentCursor;
    bool cursorVisible = true;
    vrml::Player *player = nullptr;
    std::list<void (*)()> playerUseList;

    int activeClippingPlane;

    osg::ref_ptr<osg::Geode> intersectedNode;
    osg::ref_ptr<osg::Drawable> intersectedDrawable;
    //osg::ref_ptr<osg::NodePath> intersectedNodePath;
    osg::NodePath intersectedNodePath;
    osg::Vec3 intersectionHitPointWorld;
    osg::Vec3 intersectionHitPointWorldNormal;
    osg::Vec3 intersectionHitPointLocal;
    osg::Vec3 intersectionHitPointLocalNormal;
    osg::ref_ptr<osg::RefMatrix> intersectionMatrix;

    mutable coPointerButton *pointerButton = nullptr;
    mutable coPointerButton *mouseButton = nullptr;
    mutable coPointerButton *relativeButton = nullptr;
    vrui::coToolboxMenu *m_toolBar = nullptr;
    vrui::coMenu *m_vruiMenu = nullptr;
    double interactorScale = 1.;

    int numClipPlanes;

    coVRPluginSupport();
    ~coVRPluginSupport();

    std::vector<std::ostream *> m_notifyStream;
    std::vector<NotifyBuf *> m_notifyBuf;
};

COVEREXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &buffer, const osg::Matrixd &matrix);
COVEREXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &buffer, osg::Matrixd &matrix);

COVEREXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &buffer, const osg::Vec3f &vec);
COVEREXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &buffer, osg::Vec3f &vec);

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

extern COVEREXPORT coVRPluginSupport *cover;
}
#endif
