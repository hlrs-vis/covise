/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _JSBSim_NODE_PLUGIN_H
#define _JSBSim_NODE_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/input/dev/Joystick/Joystick.h>

#include <util/common.h>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <cfenv>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRNavigationManager.h>
#include <config/CoviseConfig.h>
#include <util/byteswap.h>
#include <net/covise_connect.h>

#include <initialization/FGTrim.h>
#include <FGFDMExec.h>
#include <input_output/FGXMLFileRead.h>

#include <util/coTypes.h>
#include <cover/ui/Button.h>
#include <cover/ui/Action.h>
#include <cover/ui/Menu.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Label.h>
#include <cover/ui/SelectionList.h>

#include <proj.h>

#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFVec3f.h>

#include <rsClient/remoteSoundClient.h>


class UDPComm;

using JSBSim::FGXMLFileRead;
using JSBSim::Element;
using namespace opencover;
using namespace covise;
using namespace vrml;

class JSBSimPlugin : public coVRPlugin, public ui::Owner, public opencover::coVRNavigationProvider
{
public:
    JSBSimPlugin();
    ~JSBSimPlugin();
    bool init();
    bool destroy();

    bool update();
    virtual void setEnabled(bool);
    static JSBSimPlugin* instance() { return plugin; };
    void addThermal(const osg::Vec3& velocity, float turbulence);

private:
    static JSBSimPlugin* plugin;
    ui::Menu* JSBMenu;
    ui::Action* printCatalog;
    ui::Button* pauseButton;
    ui::Button* DebugButton;
    ui::Action* resetButton;
    ui::Action* upButton;
    ui::Group* Weather;
    ui::Group* Geometry;
    ui::Label* WindLabel;
    ui::Label* VLabel;
    ui::Label* VzLabel;
    ui::EditField* WX;
    ui::EditField* WY;
    ui::EditField* WZ;
    ui::EditField* tX;
    ui::EditField* tY;
    ui::EditField* tZ;
    ui::EditField* tH;
    ui::EditField* tP;
    ui::EditField* tR;
    ui::EditField* tS;
    ui::SelectionList* planeType;
    std::unique_ptr<config::Array<std::string>> aircrafts;
    remoteSound::Client* rsClient;
    void initAircraft();

    Joystick* joystickDev = nullptr;
    //bool state0 = false;
    //bool state1 = false;
    //bool state2 = false;

    SGPath RootDir;
    SGPath ScriptName;
    std::string AircraftDir;
    std::string EnginesDir;
    std::string SystemsDir;
    std::string AircraftName;
    std::string resetFile;
    std::string geometryFile;
    std::string currentAircraft;
    osg::ref_ptr<osg::MatrixTransform> geometryTrans;
    vector <string> LogOutputName;
    vector <SGPath> LogDirectiveName;
    vector <string> CommandLineProperties;
    vector <double> CommandLinePropertyValues;
    osg::Matrix eyePoint;

    double current_seconds = 0.0;
    double SimStartTime = 0.0;
    double frame_duration = 0.0;
    double printTime = 0.0;
    std::unique_ptr<config::Value<double>> cX;
    std::unique_ptr<config::Value<double>> cY;
    std::unique_ptr<config::Value<double>> cZ;
    std::unique_ptr<config::Value<double>> cH;
    std::unique_ptr<config::Value<double>> cP;
    std::unique_ptr<config::Value<double>> cR;
    std::unique_ptr<config::Value<double>> cS;

    JSBSim::FGFDMExec* FDMExec = nullptr;
    JSBSim::FGTrim* trimmer = nullptr;

    std::shared_ptr <JSBSim::FGAtmosphere>      Atmosphere;
    std::shared_ptr <JSBSim::FGWinds>           Winds;
    std::shared_ptr <JSBSim::FGFCS>            FCS;
    std::shared_ptr <JSBSim::FGPropulsion>      Propulsion;
    std::shared_ptr <JSBSim::FGMassBalance>     MassBalance;
    std::shared_ptr <JSBSim::FGAircraft>        Aircraft;
    std::shared_ptr <JSBSim::FGPropagate>       Propagate;
    std::shared_ptr <JSBSim::FGAuxiliary>       Auxiliary;
    std::shared_ptr <JSBSim::FGAerodynamics>    Aerodynamics;
    std::shared_ptr <JSBSim::FGGroundReactions> GroundReactions;
    std::shared_ptr <JSBSim::FGInertial>        Inertial;
    std::shared_ptr <JSBSim::FGAccelerations>   Accelerations;

    JSBSim::FGPropagate::VehicleState initialLocation;

    osg::Vec3d zeroPosition;

    JSBSim::FGLocation il;

#pragma pack(push, 1)
    struct FGControl
    {
        double elevator;
        double aileron;
        double throttle;
    } fgcontrol;
    struct GliderValues
    {
	int32_t left;
	int32_t right;
	int32_t angle;
	int32_t speed;
	uint32_t state;
    };
    GliderValues gliderValues;
#pragma pack(pop)
    UDPComm* udp;
    int deviceVersion=1;
    void initUDP();
    bool initJSB();
    bool updateUdp();
    void reset(double dz = 0.0);
    void updateTrans();

    //! this functions is called when a key is pressed or released
    virtual void key(int type, int keySym, int mod);

    bool realtime;
    bool play_nice;
    bool suspend;
    bool catalog;
    bool nohighlight;

    double end_time = 1e99;
    double actual_elapsed_time = 0;
    double cycle_duration = 0;
    OpenThreads::Mutex mutex;
    PJ* coordTransformation;
    std::string coviseSharedDir;
    osg::Vec3d projectOffset;
    osg::Matrix lastPos;
    osg::Vec3 currentVelocity;
    float currentTurbulence;
    osg::Vec3 targetVelocity;
    float targetTurbulence;
    remoteSound::Sound* varioSound;
    remoteSound::Sound* windSound;
    std::string remoteSoundServer;
    int remoteSoundPort;
    std::string VarioSound;
    std::string WindSound;
    std::string host;
    unsigned short serverPort;
    unsigned short localPort;
    int Joysticknumber = -1;
    int Ruddernumber = -1;
};

class PLUGINEXPORT VrmlNodeThermal : public VrmlNodeChild
{
public:

    static void initFields(VrmlNodeThermal* node, VrmlNodeType* t);
    static const char* name();

    VrmlNodeThermal(VrmlScene* scene = 0);
    VrmlNodeThermal(const VrmlNodeThermal& n);
    virtual ~VrmlNodeThermal();

    void eventIn(double timeStamp, const char* eventName,
        const VrmlField* fieldValue);

    virtual void render(Viewer*);

    VrmlSFVec3f d_direction;
    VrmlSFVec3f d_location;
    VrmlSFFloat d_maxBack;
    VrmlSFFloat d_maxFront;
    VrmlSFFloat d_minBack;
    VrmlSFFloat d_minFront;
    VrmlSFFloat d_height;
    VrmlSFVec3f d_velocity;
    VrmlSFFloat d_turbulence;
    static int numThermalNodes;
private:

};
#endif
