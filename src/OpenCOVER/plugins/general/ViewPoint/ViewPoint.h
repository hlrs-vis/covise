/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <vector>
#include <cover/coVRPluginSupport.h>

#include <cover/coVRPlugin.h>
#include <OpenVRUI/osg/mathUtils.h>
#include "FlightPathVisualizer.h"
#include "Interpolator.h"

#include <cover/ui/Owner.h>


namespace opencover {
namespace ui {
class Menu;
class Action;
class Button;
class Slider;
}
}

class ViewDesc;

using namespace opencover;

const int SHARED_VP_NAME_LENGTH = 1024;

typedef struct SharedActiveVPData
{
    int totNum;
    int isEnabled;
    int index;
    char name[SHARED_VP_NAME_LENGTH];
    int hasClipPlane[6];
} SharedActiveVPData;

class ViewPoints : public coVRPlugin, public ui::Owner
{
    friend class ViewDesc;
public:
    // create pinboard button viewpoints and submenu
    ViewPoints();
    ~ViewPoints() override;

    static SharedActiveVPData *actSharedVPData;
    
    bool init() override;
    bool init2() override;
    bool update() override;
    void preFrame() override;
    void addNode(osg::Node *, const RenderObject *) override;
    void key(int type, int keySym, int mod) override;
    void guiToRenderMsg(const char *msg) override;
    void message(int toWhom, int type, int length, const void *data) override;
    virtual ui::Action *getMenuButton(const std::string &buttonName);

    void readFromDom();
    void saveAllViewPoints();

    ViewDesc *activeVP = nullptr;
    coCoord curr_coord;
    osg::Matrix currentMat_;
    float curr_scale;
    double initTime;
    double pauseTime = -1.;
    bool moveToFlightTime = false;
    bool flyingStatus = false;
    bool runFlight = false;
    bool flyingMode;
    int flight_index;
    float flightTime;
    int vp_index;
    bool vp_index_changed = false;
    bool isQuickNavEnabled;

    void saveViewPoint(const char *suggestedName = NULL);
    void changeViewPoint(const char *name);
    void changeViewPoint(int id);
    void changeViewPointName(int id, const char *newName);
    void createViewPoint(const char *name, int guiId, const char *desc, const char *plane);
    void deleteViewPoint(int id);

    // fly around all items marked in flight list
    void completeFlight(bool state);
    void playOrPauseFlight(bool state);
    void loadViewpoint(int id);
    void loadViewpoint(const char *name);
    void loadViewpoint(ViewDesc *viewDesc);
    void activateViewpoint(ViewDesc *viewDesc);
    void startTurnTableAnimation(float time);
    void turnTableStep();

    void nextViewpoint();
    void previousViewpoint();

    int numEnabledViewpoints() const;

    void enableHUD();
    void disableHUD();
    void setDataChanged(){dataChanged = true;};

    static bool showInteractors;
    static bool showFlightpath;
    static bool showCamera;
    static bool shiftFlightpathToEyePoint;
    static bool showViewpoints;

    static ViewDesc *lastVP;

    static Vec3 tangentOut;
    static Vec3 tangentIn;

    static Interpolator::EasingFunction curr_easingFunc;
    static Interpolator::TranslationMode curr_translationMode;
    static Interpolator::RotationMode curr_rotationMode;

    ui::Menu *editVPMenu_ = nullptr;
    ui::Button *showFlightpathCheck_ = nullptr;
    ui::Button *showCameraCheck_ = nullptr;

    ui::Button *shiftFlightpathCheck_ = nullptr;

    FlightPathVisualizer *flightPathVisualizer;
    Interpolator *vpInterpolator;

    static ViewPoints *instance(){return inst;}
    bool dataChanged = false;
    bool isClipPlaneChecked();

private:
    Vec3 eyepoint;

    int id_;
    int fileNumber;
    int frameNumber;
    int numberOfDefaultVP;
    float *positions;
    float *orientations;
    std::vector<osg::Vec3> maxPos;
    std::vector<osg::Quat> maxQuat;
    bool record;
    bool activated_; ///< whether we have once
    std::string vwpPath;
    bool videoBeingCaptured;

    FILE *fp;

    ui::Menu *viewPointMenu_; //< menu for View Points
    ui::Menu *flightMenu_; //< menu for Flight
    ui::Action *saveButton_;
    ui::Button *flyingModeCheck_;
    ui::Button *runButton;
    ui::Slider *animPositionSlider;
    ui::Button *useClipPlanesCheck_;
    ui::Button *turnTableAnimationCheck_;
    ui::Button *showAvatar_;

    osg::ref_ptr<osg::Geode> qnNode;
    std::vector<ViewDesc *> viewpoints; //< list of viewpoints

    void updateViewPointIndex();

    bool nextFlightIndex();

    void stopRecord();
    void startRecord();

    void updateSHMData();

    void sendDefaultViewPoint();
    void sendNewViewpointMsgToGui(const char *name, int index, const char *str, const char *plane);
    void sendNewDefaultViewpointMsgToGui(const char *name, int index);
    void sendViewpointChangedMsgToGui(const char *name, int index, const char *str, const char *plane);
    void sendLoadViewpointMsgToGui(int index);
    void sendFlyingModeToGui();
    void sendChangeIdMsgToGui(int guiId, int newId);


    // starts search at the node, searches for string...
    // returns the found nodes in the following array
    void addNodes(osg::Node *n, std::string s, std::vector<osg::Node *> &nodes);
    bool isOn(const char *vp);

    void changeViewDesc(ViewDesc *viewDesc);
    void changeViewDesc(Matrix newMatrix, float newScale, Vec3 newTanIn, Vec3 newTanOut, int idd, const char *name, ViewDesc *currVP);

    void alignViewpoints(char alignment);
    void setCatmullRomTangents();
    void setStraightTangents();
    void setEqualTangents();

    float turnTableViewingAngle_;

    bool turnTableAnimation_;
    double turnTableAnimationTime_;
    float turnTableCurrent_;

    float turnTableStepAlpha_;
    bool turnTableStep_;
    double turnTableStepInitTime_;
    osg::Matrix turnTableStepInitMat_, turnTableStepDestMat_;
    static ViewPoints *inst;
    bool loopMode = false;
    bool sendActivatedViewpointMsg = false;
};
