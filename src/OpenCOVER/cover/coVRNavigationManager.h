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

#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>

#include <osg/Vec3>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Geometry>

namespace vrui
{
class coNavInteraction;
class coTrackerButtonInteraction;
class coMouseButtonInteraction;
}
namespace opencover
{
class coMeasurement;
class buttonSpecCell;
class coVRLabel;
class COVEREXPORT coVRNavigationManager : public vrui::coMenuListener
{
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
        Measure
    };

    float AnalogX, AnalogY;

    coVRNavigationManager();

    ~coVRNavigationManager();

    static coVRNavigationManager *instance();
    static void xformCallback(void *mgr, buttonSpecCell *spec);
    static void xformRotateCallback(void *mgr, buttonSpecCell *spec);
    static void xformTranslateCallback(void *mgr, buttonSpecCell *spec);
    static void scaleCallback(void *mgr, buttonSpecCell *spec);
    static void collideCallback(void *mgr, buttonSpecCell *spec);
    static void walkCallback(void *mgr, buttonSpecCell *spec);
    static void driveCallback(void *mgr, buttonSpecCell *spec);
    static void flyCallback(void *mgr, buttonSpecCell *spec);
    static void driveSpeedCallback(void *mgr, buttonSpecCell *spec);
    static void snapCallback(void *mgr, buttonSpecCell *spec);
    static void showNameCallback(void *mgr, buttonSpecCell *spec);
    static void measureCallback(void *mgr, buttonSpecCell *spec);
    static void traverseInteractorsCallback(void *mgr, buttonSpecCell *spec);
    static void menuCallback(void *mgr, buttonSpecCell *spec);

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
    void setOldNavMode()
    {
        setNavMode(oldNavMode);
    };
    NavMode getMode()
    {
        return navMode;
    }
    void setMode(NavMode mode);
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

    void toggleXform(bool state);
    void toggleXformRotate(bool state);
    void toggleXformTranslate(bool state);
    void toggleScale(bool state);
    void toggleFly(bool state);
    void toggleWalk(bool state);
    void toggleGlide(bool state);
    void toggleCollide(bool state);
    void toggleAxis(bool state);
    void toggleShowName(bool state);
    void toggleMeasure(bool state);
    void toggleInteractors(bool state);
    void toggleMenu();

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
    void setShowName(bool on)
    {
        toggleShowName(on);
    }

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

    osg::Vec3 rotPointVec;
    osg::ref_ptr<osg::MatrixTransform> rotPoint;

    std::vector<coMeasurement *> measurements;

    void initInteractionDevice();
    void initAxis();
    void initHandDeviceGeometry();
    void initCollMenu();
    void initMatrices();
    void initShowName();
    void initMeasure();

    virtual void menuEvent(vrui::coMenuItem *);

    void enableAllNavigations(bool enable);
};
}
#endif
