/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVR_NAVIGATION_MANAGER_H
#define COVR_NAVIGATION_MANAGER_H

/*! \file
 \brief  mouse and tracker 3D navigation

 \author
 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#include <util/common.h>

#include "ui/Owner.h"
namespace vive {
namespace ui {
class Group;
class Menu;
class Action;
class ButtonGroup;
class Button;
class Slider;
}
}

#include <vsg/maths/vec3.h>
#include <vsg/maths/mat4.h>
#include <vsg/nodes/MatrixTransform.h>

namespace vrui
{
class coNavInteraction;
class coMouseButtonInteraction;
class coRelativeInputInteraction;
class coButtonMenuItem;
class coRowMenu;
}
namespace vive
{
class vvMeasurement;
class vvLabel;
class vvPlugin;

class VVCORE_EXPORT coVRNavigationProvider
{
public:
    coVRNavigationProvider(const std::string name, vvPlugin* plugin);
    virtual ~coVRNavigationProvider();
    vvPlugin *plugin;
    ui::Button* navMenuButton = nullptr;
    std::string getName() { return name; };
    virtual void setEnabled(bool enabled);
    bool isEnabled() { return enabled; };
    int ID;
private:
    std::string name;
    bool enabled=false;
};

class VVCORE_EXPORT vvNavigationManager: public ui::Owner
{
    static vvNavigationManager *s_instance;
    vvNavigationManager();
    using NodePath = std::vector<const vsg::Node*>;

public:
    enum NavMode
    {
        NavOther=-1,
        NavNone,
        XForm,
        Scale,
        Fly,
        Glide,
        Walk,
        ShowName,
        XFormRotate,
        XFormTranslate,
        TraverseInteractors,
        Menu,
        Measure,
        Select,
        SelectInteract,
        NumNavModes // keep last
    };

    double AnalogX, AnalogY;

    ~vvNavigationManager();
    void updatePerson();

    static vvNavigationManager *instance();

    // process key events
    bool keyEvent(vsg::KeyPressEvent& keyPress);

    void doWalkMoveToFloor();
    void processHotKeys(int keymask);
    void adjustFloorHeight();
    bool getCollision()
    {
        return collision;
    }

    void update();
    void setMenuMode(bool state);

    void updateHandMat(vsg::dmat4 &mat);
    void setHandType(int pt);
    void setNavMode(NavMode mode, bool updateGroup=true);
    void setNavMode(std::string navMode);
    NavMode getMode()
    {
        return navMode;
    }
    ui::ButtonGroup *navGroup() const;
    bool isNavigationEnabled()
    {
        return navMode != NavNone;
    }
    bool mouseNav()
    {
        return doMouseNav;
    }
    bool isViewerPosRotationEnabled()
    {
        return isViewerPosRotation;
    }
    void enableViewerPosRotation(bool b)
    {
        isViewerPosRotation = b;
    }

    void saveCurrentBaseMatAsOldBaseMat();
    bool avoidCollision(vsg::vec3 &glideVec); // returns true if collision occurred

    void wasJumping();
    float getDriveSpeed();
    void setDriveSpeed(float speed);
    bool isSnapping() const;
    bool isDegreeSnapping() const;
    float snappingDegrees() const;
    void enableSnapping(bool enable);
    void enableDegreeSnapping(bool enable, float degree);
    bool restrictOn() const;

    void setStepSize(float stepsize);
    float getStepSize() const;
    void doGuiScale(float scale);
    void doGuiRotate(float x, float y, float z);
    void doGuiTranslate(float x, float y, float z);

    int readConfigFile();

    void toggleShowName(bool state);
    void toggleInteractors(bool state);
    void toggleCollide(bool state);

    void startXform();
    void doXform();
    void stopXform();
    void startScale();
    void doScale();
    void stopScale();
    void startWalk();
    void doWalk();
    void stopWalk();
    void startDrive();
    void doDrive();
    void stopDrive();
    void startFly();
    void doFly();
    void stopFly();
    void doMouseFly();
    void doMouseXform();
    void doMouseScale();
    void doMouseScale(float);
    void doMouseWalk();
    void stopMouseNav();
    void startMouseNav();
    void startShowName();
    void doShowName();
    void stopShowName();
    void startMeasure();
    void doMeasure();
    void stopMeasure();

    void toggleSelectInteract(bool state);
    void startSelectInteract();
    void doSelectInteract();
    void stopSelectInteract(bool mouse);


    void doXformRotate();
    void doXformTranslate();

    void highlightSelectedNode(vsg::Node *selectedNode);
    double speedFactor(double delta) const;
    vsg::dvec3 applySpeedFactor(vsg::dvec3 vec) const;

    void getHandWorldPosition(float *, float *, float *);

    float getPhi(float relCoord1, float width1);

    float getPhiZHori(float x2, float x1, float y2, float widthY, float widthX);
    float getPhiZVerti(float y2, float y1, float x2, float widthX, float widthY);
    void makeRotate(float heading, float pitch, float roll, int headingBool, int pitchBool, int rollBool);

    void disableRotationPoint()
    {
        rotationPoint = false;
    }
    void disableRotationAxis()
    {
        rotationAxis = false;
    }
    bool getRotationPointActive()
    {
        return rotationPoint;
    }
    bool getRotationAxisAcitve()
    {
        return rotationAxis;
    }
    void setRotationPoint(float x, float y, float z, float size = 1.f);
    void setRotationPointVisible(bool visible);
    vsg::dvec3 getRotationPoint()
    {
        return rotPointVec;
    }
    void setRotationAxis(float x, float y, float z);
    void setTranslateFactor(float f)
    {
        guiTranslateFactor = f;
    }
    void registerNavigationProvider(coVRNavigationProvider*);
    void unregisterNavigationProvider(coVRNavigationProvider*);


private:
    bool doMouseNav;
    int mouseNavButtonRotate, mouseNavButtonScale, mouseNavButtonTranslate;
    bool wiiNav;
    double menuButtonStartTime;
    double menuButtonQuitInterval;

    bool collision;
    bool ignoreCollision;
    NavMode navMode;
    NavMode oldNavMode;
    std::list<coVRNavigationProvider*> navigationProviders;

    /* until button is released */

    bool shiftEnabled, shiftMouseNav;
    bool isViewerPosRotation; // mouse rotate around current viewer position
    vsg::dmat4 mat0;

    vsg::dmat4 invBaseMatrix;
    vsg::dmat4 oldInvBaseMatrix;
    vsg::Node *oldFloorNode = nullptr;
	vsg::dmat4 oldFloorMatrix;
	NodePath oldNodePath;


    float currentVelocity;

    vsg::dmat4 old_mat, old_dcs_mat;
    vsg::dmat4 old_xform_mat;
    vsg::dmat4 handMat;
    vsg::dvec3 startHandPos; //we need the start value for ongoing interaction
    vsg::dvec3 startHandDir; //we need the start value for ongoing interaction
    vsg::dvec3 handPos, oldHandPos;
    vsg::dvec3 handDir, oldHandDir;
    vsg::dvec3 transformVec;
    vsg::dvec3 rotationVec;
    bool rotationPoint;
    bool rotationPointVisible;
    bool rotationAxis;
    float guiTranslateFactor;

    float actScaleFactor; //fuer die Skalieroption, Initialisierung in der naechsten ifSchleife
    float mx, my;
    float x0, y0, relx0, rely0;
    float oldRotx, newRotx, oldRoty, newRoty;
    float modifiedVSize, modifiedHSize, yValViewer, yValObject;
    vsg::vec3 transRel;
    float originX, originY;

    int wiiFlag;

    vsg::Node *oldSelectedNode_;

    float oldDcsScaleFactor;

    vsg::dvec3 currentLeftPos; // in WC
    vsg::dvec3 currentRightPos;
    vsg::dvec3 oldLeftPos; // in WC
    vsg::dvec3 oldRightPos;
    float collisionDist;

    int jsZeroPosX, jsZeroPosY, jsOffsetX, jsOffsetY, jsXmax, jsYmax, jsXmin, jsYmin;
    bool jsEnabled;

    bool visensoJoystick;
    bool joystickActive;

    vrui::coNavInteraction *interactionA = nullptr; ///< interaction for first button
    vrui::coNavInteraction *interactionB = nullptr; ///< interaction for second button
    vrui::coNavInteraction *interactionC = nullptr; ///< interaction for third button
    vrui::coNavInteraction *interactionMenu = nullptr; ///< interaction for steadycam
    vrui::coNavInteraction *interactionShortcut = nullptr; ///< interaction for navigation with keyboard shortcuts
    vrui::coMouseButtonInteraction *interactionMA = nullptr; ///< interaction for first mouse button
    vrui::coMouseButtonInteraction *interactionMB = nullptr; ///< interaction for first mouse button
    vrui::coMouseButtonInteraction *interactionMC = nullptr; ///< interaction for first mouse button
    vrui::coRelativeInputInteraction *interactionRel = nullptr; ///< interaction for relative input (space mouse) without button

    double navExp;

    double syncInterval;

    double stepSize;

    double driveSpeed;

    void init();
    bool jump; // set to true if a jump has been performed to disable collision detection
    bool snapping;
    bool snappingD;
    double snapDegrees;
    bool m_restrict = false;
    double rotationSpeed;
    bool turntable;
    bool animationWasRunning=false;

    bool showGeodeName_;
    vsg::Node *oldShowNamesNode_ = nullptr;
    vvLabel *nameLabel_;
    vrui::coRowMenu *nameMenu_;
    vrui::coButtonMenuItem *nameButton_;
    ui::Menu *navMenu_ = nullptr;
    ui::Action *m_viewAll=nullptr, *m_resetView=nullptr, *m_viewVisible=nullptr;
    ui::Group *navModes_ = nullptr;
    ui::ButtonGroup *navGroup_ = nullptr;
    ui::Button *noNavButton_=nullptr;
    ui::Button *xformButton_=nullptr, *scaleButton_=nullptr, *flyButton_=nullptr, *walkButton_=nullptr, *driveButton_=nullptr;
    ui::Button *xformRotButton_=nullptr, *xformTransButton_=nullptr, *selectButton_=nullptr, *showNameButton_=nullptr;
    ui::Button *selectInteractButton_=nullptr;
    ui::Button *measureButton_=nullptr, *traverseInteractorButton_=nullptr;
    ui::Button *collisionButton_=nullptr, *snapButton_=nullptr;
    ui::Slider *driveSpeedSlider_=nullptr;
    ui::Action *scaleUpAction_=nullptr, *scaleDownAction_=nullptr;
    ui::Slider *scaleSlider_=nullptr;
    ui::Action *centerViewButton = nullptr;
    ui::Action *printObjectTransform = nullptr;
    vsg::dvec3 rotPointVec;
    vsg::ref_ptr<vsg::MatrixTransform> rotPoint;

    std::vector<vvMeasurement *> measurements;

    void initInteractionDevice();
    void initAxis();
    void initHandDeviceGeometry();
    void initCollMenu();
    void initMatrices();
    void initMenu();
    void initShowName();
    void initMeasure();

    vsg::dvec3 getCenter() const;
    void centerView();
    vsg::dvec3 mouseNavCenter;
};
}
#endif
