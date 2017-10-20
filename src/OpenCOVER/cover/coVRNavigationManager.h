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
namespace opencover {
namespace ui {
class Menu;
class Action;
class ButtonGroup;
class Button;
class Slider;
}
}

#include <osg/Vec3>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Geometry>

namespace vrui
{
class coNavInteraction;
class coMouseButtonInteraction;
class coButtonMenuItem;
class coRowMenu;
}
namespace opencover
{
class coMeasurement;
class coVRLabel;

class COVEREXPORT coVRNavigationManager: public ui::Owner
{
    static coVRNavigationManager *s_instance;
    coVRNavigationManager();

public:
    enum NavMode
    {
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
        Select
    };

    float AnalogX, AnalogY;

    ~coVRNavigationManager();
    void updatePerson();

    static coVRNavigationManager *instance();

    // process key events
    bool keyEvent(int type, int keySym, int mod);
    bool mouseEvent(int type, int state, int code);

    void doWalkMoveToFloor();
    void processHotKeys(int keymask);
    void adjustFloorHeight();
    bool getCollision()
    {
        return collision;
    }

    void update();
    void setMenuMode(bool state);

    void updateHandMat(osg::Matrix &mat);
    void setHandType(int pt);
    void setNavMode(NavMode mode);
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
    bool avoidCollision(osg::Vec3 &glideVec); // returns true if collision occurred

    void wasJumping();
    float getDriveSpeed();
    void setDriveSpeed(float speed);
    bool isSnapping() const;
    bool isDegreeSnapping() const;
    float snappingDegrees() const;
    void enableSnapping(bool enable);
    void enableDegreeSnapping(bool enable, float degree);

    void setStepSize(float stepsize);
    float getStepSize() const;
    void doGuiScale(float scale);
    void doGuiRotate(float x, float y, float z);
    void doGuiTranslate(float x, float y, float z);

    int readConfigFile();

    void toggleShowName(bool state);
    void toggleInteractors(bool state);
    void toggleCollide(bool state);
#if 0
    void toggleXform(bool state);
    void toggleXformRotate(bool state);
    void toggleXformTranslate(bool state);
    void toggleScale(bool state);
    void toggleFly(bool state);
    void toggleWalk(bool state);
    void toggleGlide(bool state);
    void toggleAxis(bool state);
    void toggleMeasure(bool state);
    void toggleMenu();
#endif

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
    void doScale(float);
    void doMouseWalk();
    void stopMouseNav();
    void startMouseNav();
    void startShowName();
    void doShowName();
    void stopShowName();
    void startMeasure();
    void doMeasure();
    void stopMeasure();

    void doXformRotate();
    void doXformTranslate();

    void highlightSelectedNode(osg::Node *selectedNode);
    double speedFactor(double delta);

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
    osg::Vec3 getRotationPoint()
    {
        return rotPointVec;
    }
    void setRotationAxis(float x, float y, float z);
    void setTranslateFactor(float f)
    {
        guiTranslateFactor = f;
    }
#if 0
    void setShowName(bool on)
    {
        toggleShowName(on);
    }
#endif

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

    /* until button is released */

    bool shiftEnabled, shiftMouseNav;
    bool isViewerPosRotation; // mouse rotate around current viewer position
    osg::Matrix mat0;

    osg::Matrix invBaseMatrix;
    osg::Matrix oldInvBaseMatrix;
	osg::Node *oldFloorNode;
	osg::Matrix oldFloorMatrix;
	osg::NodePath oldNodePath;


    float currentVelocity;

    osg::Matrix old_mat, old_dcs_mat;
    osg::Matrix old_xform_mat;
    osg::Matrix handMat;
    osg::Vec3 startHandPos; //we need the start value for ongoing interaction
    osg::Vec3 startHandDir; //we need the start value for ongoing interaction
    osg::Vec3 handPos, oldHandPos;
    osg::Vec3 handDir, oldHandDir;
    osg::Vec3 transformVec;
    osg::Vec3 rotationVec;
    bool rotationPoint;
    bool rotationPointVisible;
    bool rotationAxis;
    float guiTranslateFactor;
    float startFrame;

    float actScaleFactor; //fuer die Skalieroption, Initialisierung in der naechsten ifSchleife
    float mx, my;
    float x0, y0, relx0, rely0;
    float oldRotx, newRotx, oldRoty, newRoty;
    float modifiedVSize, modifiedHSize, yValViewer, yValObject;
    float transXRel, transYRel, transZRel;
    float originX, originY;
    int curTypeYRot;
    int curTypeZRot;
    int curTypeDef;
    int curTypeZTrans;
    int curTypeContRot;

    int wiiFlag;

    osg::Node *oldSelectedNode_;

    float oldDcsScaleFactor;

    osg::Vec3 currentLeftPos; // in WC
    osg::Vec3 currentRightPos;
    osg::Vec3 oldLeftPos; // in WC
    osg::Vec3 oldRightPos;
    float collisionDist;

    int jsZeroPosX, jsZeroPosY, jsOffsetX, jsOffsetY, jsXmax, jsYmax, jsXmin, jsYmin;
    bool jsEnabled;

    bool visensoJoystick;
    bool joystickActive;

    vrui::coNavInteraction *interactionA; ///< interaction for first button
    vrui::coNavInteraction *interactionB; ///< interaction for second button
    vrui::coNavInteraction *interactionC; ///< interaction for third button
    vrui::coNavInteraction *interactionMenu; ///< interaction for steadycam
    vrui::coMouseButtonInteraction *interactionMA; ///< interaction for first mouse button
    vrui::coMouseButtonInteraction *interactionMB; ///< interaction for first mouse button
    vrui::coMouseButtonInteraction *interactionMC; ///< interaction for first mouse button

    float navExp;

    float syncInterval;

    float stepSize;

    float driveSpeed;

    void init();
    bool navigating;
    bool jump; // set to true if a jump has been performed to disable collision detection
    bool snapping;
    bool snappingD;
    float snapDegrees;
    float rotationSpeed;
    int oldKeyMask;
    bool turntable;

    bool showNames_;
    bool showGeodeName_;
    osg::Node *oldShowNamesNode_;
    coVRLabel *nameLabel_;
    vrui::coRowMenu *nameMenu_;
    vrui::coButtonMenuItem *nameButton_;
    ui::Menu *navMenu_ = nullptr;
    ui::ButtonGroup *navGroup_ = nullptr;
    ui::Button *xformButton_=nullptr, *scaleButton_=nullptr, *flyButton_=nullptr, *walkButton_=nullptr, *driveButton_=nullptr;
    ui::Button *xformRotButton_=nullptr, *xformTransButton_=nullptr, *selectButton_=nullptr, *showNameButton_=nullptr;
    ui::Button *measureButton_=nullptr, *traverseInteractorButton_=nullptr;
    ui::Button *collisionButton_=nullptr, *snapButton_=nullptr;
    ui::Slider *driveSpeedSlider_=nullptr;

    osg::Vec3 rotPointVec;
    osg::ref_ptr<osg::MatrixTransform> rotPoint;

    std::vector<coMeasurement *> measurements;

    void initInteractionDevice();
    void initAxis();
    void initHandDeviceGeometry();
    void initCollMenu();
    void initMatrices();
    void initMenu();
    void initShowName();
    void initMeasure();

#ifdef VRUI
    virtual void menuEvent(vrui::coMenuItem *);
#endif
};
}
#endif
