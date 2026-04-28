/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _JSBSim_NODE_PLUGIN_H
#define _JSBSim_NODE_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/input/dev/Joystick/Joystick.h>

#include <string>
#include <util/common.h>

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <cfenv>
#include <map>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRNavigationManager.h>
#include <config/CoviseConfig.h>
#include <util/byteswap.h>
#include <net/covise_connect.h>
#include <audio/Audio.h>
#include <audio/Player.h>

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

class UDPComm;

using JSBSim::Element;
using JSBSim::FGXMLFileRead;
using namespace opencover;
using namespace covise;

struct AircraftInfo
{
    std::string name;
    std::string displayName;
    std::string geometryFile;
    std::string systemsDir;
    std::string enginesDir;
    osg::Matrix geometryTransform;
};

enum JoystickActionType
{
    AILERON,
    ELEVATOR,
    RUDDER,
    THROTTLE,
    MIXTURE
};
class JoystickAction
{
public:
    int joystickNumber;
    JoystickActionType type;

    int axisNumber = -1;
    int sliderNumber = -1;
    int engine = -1;
    bool invert = false;

    void update(Joystick *device)
    {
        m_oldValue = m_value;

        if (axisNumber >= 0)
        {
            m_value = device->axes[joystickNumber][axisNumber];
        }
        if (sliderNumber >= 0)
        {
            m_value = device->sliders[joystickNumber][sliderNumber];
        }
    }
    bool getChangedValue(double &value) const
    {
        if (m_oldValue != m_value)
        {
            value = invert ? -m_value : m_value;
            return true;
        }
        return false;
    }

private:
    double m_value = 0.0;
    double m_oldValue = 0.0;
};

class JSBSimPlugin : public coVRPlugin, public ui::Owner, public opencover::coVRNavigationProvider
{
public:
    JSBSimPlugin();
    ~JSBSimPlugin();

    static JSBSimPlugin *instance() { return plugin; };

    bool init();
    bool update();

    // from coVRNavigationProvider
    virtual void setEnabled(bool) override;

    void addThermal(const osg::Vec3 &velocity, float turbulence);

private:
    osg::Vec3f getOriginOffset() const;

    void readJoystickConfiguration();
    void loadAvailableAircraft();
    void loadAircraft(const std::string &aircraftName);

    void updateInputs();

    void windChangedCallback();

    static JSBSimPlugin *plugin;
    ui::Menu *JSBMenu;
    ui::Action *printCatalog;
    ui::Button *pauseButton;
    ui::Button *debugButton;
    ui::Action *resetButton;
    ui::Action *upButton;
    ui::Group *weatherGroup;
    ui::Label *windLabel;
    ui::Label *labelVelocityX;
    ui::Label *labelVelocityY;
    ui::Label *labelVelocityZ;
    ui::EditField *WX;
    ui::EditField *WY;
    ui::EditField *WZ;
    ui::SelectionList *planeType;
    std::map<std::string, AircraftInfo> m_availableAircraft;

    std::vector<JoystickAction> m_joystickActions;

    Joystick *joystickDev = nullptr;
    // bool state0 = false;
    // bool state1 = false;
    // bool state2 = false;

    std::string m_defaultAircraft;

    SGPath RootDir;
    SGPath ScriptName;
    std::string AircraftName;
    std::string resetFile;
    AircraftInfo *currentAircraft = nullptr;
    osg::ref_ptr<osg::MatrixTransform> geometryTrans;
    vector<string> LogOutputName;
    vector<SGPath> LogDirectiveName;
    vector<string> CommandLineProperties;
    vector<double> CommandLinePropertyValues;
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

    JSBSim::FGFDMExec *FDMExec = nullptr;
    JSBSim::FGTrim *trimmer = nullptr;
    std::string jsName;
    std::string jsThrottleName;
    std::string rudderName;

    std::shared_ptr<JSBSim::FGAtmosphere> Atmosphere;
    std::shared_ptr<JSBSim::FGWinds> Winds;
    std::shared_ptr<JSBSim::FGFCS> FCS;
    std::shared_ptr<JSBSim::FGPropulsion> Propulsion;
    std::shared_ptr<JSBSim::FGMassBalance> MassBalance;
    std::shared_ptr<JSBSim::FGAircraft> Aircraft;
    std::shared_ptr<JSBSim::FGPropagate> Propagate;
    std::shared_ptr<JSBSim::FGAuxiliary> Auxiliary;
    std::shared_ptr<JSBSim::FGAerodynamics> Aerodynamics;
    std::shared_ptr<JSBSim::FGGroundReactions> GroundReactions;
    std::shared_ptr<JSBSim::FGInertial> Inertial;
    std::shared_ptr<JSBSim::FGAccelerations> Accelerations;

    JSBSim::FGPropagate::VehicleState initialLocation;

    osg::Vec3d zeroPosition;

    JSBSim::FGLocation il;

#pragma pack(push, 1)
    struct Controls
    {
        double elevator;
        double aileron;
        double rudder;
        double throttle;
        double mixture;
    } m_controls;

    struct GliderValues
    {
        int32_t left;
        int32_t right;
        int32_t angle;
        int32_t speed;
        uint32_t state;
    };
#pragma pack(pop)

    UDPComm *udp;
    int deviceVersion = 1;
    bool initJSB();
    void updateUdp();
    void reset(double dz = 0.0);

    //! this functions is called when a key is pressed or released
    virtual void key(int type, int keySym, int mod);

    OpenThreads::Mutex mutex;
    PJ *coordTransformation;

    osg::Matrix lastPos;

    // For wind/turbulence
    osg::Vec3 m_windVelocitySetting;
    osg::Vec3 m_windVelocityCurrent;
    osg::Vec3 m_windVelocityTarget;
    float m_windTurbulenceCurrent;
    float m_windTurbulenceTarget;

    // Audio related
    audio::Audio engineAudio;
    audio::Audio varioAudio;
    audio::Audio windAudio;
    std::shared_ptr<audio::Source> engineSource;
    std::shared_ptr<audio::Source> varioSource;
    std::shared_ptr<audio::Source> windSource;

    std::string host;
    unsigned short serverPort;
    unsigned short localPort;
};

#endif
