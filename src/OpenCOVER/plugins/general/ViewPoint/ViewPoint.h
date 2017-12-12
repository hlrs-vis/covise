/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <vector>
#include <cover/coVRPluginSupport.h>

using namespace vrui;
using namespace opencover;

#include <cover/coVRPlugin.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <OpenVRUI/coMenuItem.h>
#include "FlightPathVisualizer.h"
#include "Interpolator.h"
//#include "FlightManager.h"
// #include "QuickNavDrawable.h"

#define MAXFILECOL 200

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coButtonMenuItem;
class coCheckboxMenuItem;
}
class ViewDesc;

const int SHARED_VP_NAME_LENGTH = 1024;

typedef struct SharedActiveVPData
{
    int totNum;
    int isEnabled;
    int index;
    char name[SHARED_VP_NAME_LENGTH];
    int hasClipPlane[6];
} SharedActiveVPData;

class ViewPoints : public coVRPlugin, public coMenuListener
{
public:
    // create pinboard button viewpoints and submenu
    ViewPoints();
    virtual ~ViewPoints();

    static SharedActiveVPData *actSharedVPData;
    
    virtual bool init();
    virtual bool init2();
    virtual void preFrame();
    virtual void addNode(osg::Node *, const RenderObject *);
    virtual void key(int type, int keySym, int mod);
    virtual void guiToRenderMsg(const char *msg);
    virtual void message(int type, int length, const void *data);
    virtual coMenuItem *getMenuButton(const std::string &buttonName);

    void readFromDom();
    void saveAllViewPoints();

    ViewDesc *activeVP;
    coCoord curr_coord;
    osg::Matrix currentMat_;
    float curr_scale;
    double initTime;
    int flyingStatus;
    bool flyingMode;
    int flight_index;
    float flightTime;
    int vp_index;
    bool isQuickNavEnabled;

    void saveViewPoint(const char *suggestedName = NULL);
    void changeViewPoint(const char *name);
    void changeViewPoint(int id);
    void changeViewPointName(int id, const char *newName);
    void createViewPoint(const char *name, int guiId, const char *desc, const char *plane);
    void deleteViewPoint(int id);

    // fly around all items marked in flight list
    void completeFlight();
    void loadViewpoint(int id);
    void loadViewpoint(const char *name);
    void loadViewpoint(ViewDesc *viewDesc);
    void startTurnTableAnimation(float time);
    void turnTableStep();

    void nextViewpoint();
    void previousViewpoint();

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

    coRowMenu *editVPMenu_;
    coCheckboxMenuItem *showFlightpathCheck_;
    coCheckboxMenuItem *showCameraCheck_;
    coCheckboxMenuItem *showViewpointsCheck_;
    coCheckboxMenuItem *showInteractorsCheck_;

    coSubMenuItem *editVPMenuButton_;
    coCheckboxMenuItem *shiftFlightpathCheck_;
    coButtonMenuItem *vpSaveButton_;
    coButtonMenuItem *vpReloadButton_;

    coButtonMenuItem *setCatmullRomButton_;
    coButtonMenuItem *setStraightButton_;
    coButtonMenuItem *setEqualTangentsButton_;
    coButtonMenuItem *alignZButton_;
    coButtonMenuItem *alignXButton_;
    coButtonMenuItem *alignYButton_;
    coSubMenuItem *alignMenuButton_;
    coRowMenu *alignMenu_; //< menu for Aligning Viewpoints

    FlightPathVisualizer *flightPathVisualizer;
    Interpolator *vpInterpolator;
    //   	FlightManager *flightmanager;
    
    static ViewPoints *instance(){return inst;};
    bool dataChanged = false;
    bool isClipPlaneChecked() {return useClipPlanesCheck_->getState();};

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
    string vwpPath;
    bool videoBeingCaptured;

    FILE *fp;

    coRowMenu *viewPointMenu_; //< menu for View Points
    coRowMenu *flightMenu_; //< menu for Flight
    coSubMenuItem *viewPointButton_;
    coSubMenuItem *flightMenuButton_;
    coButtonMenuItem *saveButton_;
    coCheckboxMenuItem *flyingModeCheck_;
    coButtonMenuItem *runButton_;
    coButtonMenuItem *startStopRecButton_;
    coCheckboxMenuItem *useClipPlanesCheck_;
    coCheckboxMenuItem *turnTableAnimationCheck_;
    coButtonMenuItem *turnTableStepButton_;

    osg::ref_ptr<osg::Geode> qnNode;
    std::vector<ViewDesc *> viewpoints; //< list of viewpoints

    void updateViewPointIndex();

    void stopRecord();
    void startRecord();
    void startStopRec();

    void updateSHMData();
    virtual void menuEvent(coMenuItem *menuItem);

    void sendDefaultViewPoint();
    void sendNewViewpointMsgToGui(const char *name, int index, const char *str, const char *plane);
    void sendNewDefaultViewpointMsgToGui(const char *name, int index);
    void sendViewpointChangedMsgToGui(const char *name, int index, const char *str, const char *plane);
    void sendLoadViewpointMsgToGui(int index);
    void sendFlyingModeToGui();
    void sendClipplaneModeToGui();
    void sendChangeIdMsgToGui(int guiId, int newId);

    // starts search at the node, seraches for string...
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
};
