/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "JSBSim.h"
#include "models/FGFCS.h"
#include "FGJSBBase.h"
#include "initialization/FGInitialCondition.h"
#include "models/FGModel.h"
#include "models/FGMassBalance.h"
#include "models/FGPropagate.h"
#include "models/FGInertial.h"
#include "math/FGLocation.h"
#include "models/FGAccelerations.h"
#include "models/FGPropulsion.h"
#include "models/FGAerodynamics.h"
#include "models/FGAircraft.h"
#include "models/atmosphere/FGWinds.h"
#include "models/propulsion/FGEngine.h"
#include "models/propulsion/FGPiston.h"
#include "cover/VRSceneGraph.h"
#include "cover/coVRCollaboration.h"
#include <algorithm>
#include <cover/coVRPluginList.h>
#include <cover/input/input.h>
#include "cover/input/deviceDiscovery.h"
#include <cstring>
#include <osg/Math>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Vec3f>
#include <plugins/general/GeoData/GeoDataLoader.h>
#include <proj.h>
#include <string>
#include <util/UDPComm.h>
#include <util/byteswap.h>
#include <util/unixcompat.h>
#include <stdlib.h>
#include <memory>

#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/Viewer.h>

#include <cover/coVRFileManager.h>
#include <osg/Vec3>

#include <cover/input/input.h>

#include "VrmlNodeThermal.h"

inline constexpr float METERS_PER_FOOT = 0.3048f;
inline constexpr float FEET_PER_INCH = 1.f / 12.f;

JSBSimPlugin *JSBSimPlugin::plugin = NULL;

JSBSimPlugin::JSBSimPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , ui::Owner("JSBSimPlugin", cover->ui)
    , coVRNavigationProvider("JSBSim", this)
{
    fprintf(stderr, "JSBSimPlugin::JSBSimPlugin\n");
    plugin = this;

    geometryTrans = new osg::MatrixTransform();

    loadAvailableAircraft();
    readJoystickConfiguration();

    audio::Player *player = cover->getPlayer();
    if (coVRMSController::instance()->isMaster() && player)
    {
        engineAudio.setURL(coVRFileManager::instance()->getName("share/covise/jsbsim/Sounds/engine.wav"));
        varioAudio.setURL(coVRFileManager::instance()->getName("share/covise/jsbsim/Sounds/vario.wav"));
        windAudio.setURL(coVRFileManager::instance()->getName("share/covise/jsbsim/Sounds/wind1.wav"));

        engineSource = player->makeSource(&engineAudio);
        varioSource = player->makeSource(&varioAudio);
        windSource = player->makeSource(&windAudio);

        engineSource->setLoop(true);
        varioSource->setLoop(true);
        windSource->setLoop(true);

        varioSource->play();
        engineSource->play();
        windSource->play();
    }

    udp = 0;
}

void JSBSimPlugin::windChangedCallback()
{
    m_windVelocitySetting.set(WX->number(), WY->number(), WZ->number());
}

void JSBSimPlugin::loadAircraft(const std::string &aircraftName)
{
    if (m_availableAircraft.find(aircraftName) == m_availableAircraft.end())
    {
        std::cerr << "Cannot load unknown aircraft: " << aircraftName << std::endl;
        return;
    }

    // This sets the
    currentAircraft = &m_availableAircraft[aircraftName];

    // Apply transformation
    geometryTrans->setMatrix(currentAircraft->geometryTransform);

    const auto &geometryFile = currentAircraft->geometryFile;

    // remove old geometries
    while (geometryTrans->getNumChildren())
    {
        geometryTrans->removeChild(0, 1);
    }

    osg::Node *n = osgDB::readNodeFile(geometryFile.c_str());

    if (n != nullptr)
    {
        n->setName(geometryFile);
        geometryTrans->addChild(n);
    }
    else
    {
        coVRFileManager::instance()->loadFile(geometryFile.c_str(), nullptr, geometryTrans);
    }

    initJSB();
}

// this is called if the plugin is removed at runtime
JSBSimPlugin::~JSBSimPlugin()
{
    fprintf(stderr, "JSBSimPlugin::~JSBSimPlugin\n");
    cover->getScene()->removeChild(geometryTrans);
    coVRNavigationManager::instance()->unregisterNavigationProvider(this);
    if (coVRMSController::instance()->isMaster())
    {
        delete FDMExec;
        delete printCatalog;
        delete debugButton;
        delete resetButton;
        delete upButton;
        delete JSBMenu;
    }
}

#include <osg/io_utils>

void JSBSimPlugin::reset(double dz)
{
    if (FDMExec == nullptr)
    {
        initJSB();
    }
    FDMExec->Setdt(1.0 / 120.0);
    frame_duration = FDMExec->GetDeltaT();

    FDMExec->Setsim_time(0.0);
    SimStartTime = cover->frameTime();

    osg::Matrix viewer = cover->getInvBaseMat();
    std::cout << "Viewer position: " << viewer.getTrans() << std::endl;
    osg::Vec3 viewerPosition = viewer.getTrans() - getOriginOffset();
    std::cout << "Viewer position (offset): " << viewerPosition << std::endl;

    // Compute the eyepoint transformation from the aircraft. The GetXYZep method returns
    // a dimension in inches, so we divide by
    osg::Vec3 aircraftEyePoint(Aircraft->GetXYZep(1), Aircraft->GetXYZep(2), Aircraft->GetXYZep(3));
    eyePoint = osg::Matrix::translate(aircraftEyePoint / 0.00254);

    std::shared_ptr<JSBSim::FGInitialCondition> IC = FDMExec->GetIC();
    if (!IC->Load(SGPath("reset00.xml")))
    {
        cerr << "Initialization unsuccessful" << endl;
    }

    // Transform the viewer coordinates to lat/lng
    PJ_COORD c;
    c.enu.e = viewerPosition[0];
    c.enu.n = viewerPosition[1];
    c.enu.u = viewerPosition[2];
    PJ_COORD c_out = proj_trans(coordTransformation, PJ_INV, c);
    IC->SetLongitudeDegIC(c_out.lp.phi);
    IC->SetLatitudeDegIC(c_out.lp.lam);

    float altitude = viewerPosition[2] + dz;
    float altitudeFt = altitude / METERS_PER_FOOT - Aircraft->GetXYZep(3) * FEET_PER_INCH;

    std::cout << "Initialize aircraft at latitude " << c_out.lp.lam << "°, longitude " << c_out.lp.phi << "°, altitude " << altitude << "m (" << altitudeFt << " ft)" << std::endl;
    IC->SetAltitudeASLFtIC(altitudeFt);
    // there is also IC->SetAltitudeAGLFtIC(...), maybe we want that?

    // Align heading to viewer, reset roll (wings level), keep pitch according to aircraft reset file
    osg::Vec3 dir(viewer(1, 0), viewer(1, 1), viewer(1, 2));
    dir.normalize();
    IC->SetPsiRadIC(atan2(dir[1], dir[0])); // heading
    IC->SetPhiRadIC(0.0); // roll

    // IC->SetPsiRadIC(0.0); // heading
    // IC->SetThetaRadIC(-20.0*DEG_TO_RAD); // pitch from reset file beause the paraglider is sensitive to start pitch

    Propagate->InitModel();
    Propagate->InitializeDerivatives();
    FDMExec->RunIC();

    // Reset wind settings
    m_windVelocitySetting.set(WX->number(), WY->number(), WZ->number());
    m_windVelocityCurrent = m_windVelocitySetting;
    m_windVelocityTarget = m_windVelocitySetting;
    m_windTurbulenceCurrent = 0.f;
    m_windTurbulenceTarget = 0.f;

    JSBSim::FGPropagate::VehicleState vehicleState = Propagate->GetVState();

    lastPos = VRSceneGraph::instance()->getTransform()->getMatrix();
}

bool JSBSimPlugin::initJSB()
{
    // *** SET UP JSBSIM *** //
    FDMExec = new JSBSim::FGFDMExec();
    FDMExec->SetRootDir(RootDir);
    FDMExec->SetAircraftPath(SGPath("aircraft"));
    FDMExec->SetEnginePath(SGPath("engine"));
    FDMExec->SetSystemsPath(SGPath("systems"));

    FDMExec->SetPropertyValue("simulation/gravitational-torque", false);
    FDMExec->SetPropertyValue("environment/config/enabled", false);
    FDMExec->SetPropertyValue("environment/atmosphere/temperature-deg-sea-level", 15.0);
    FDMExec->SetPropertyValue("environment/atmosphere/pressure-inhg-sea-level", 29.92126);
    FDMExec->SetPropertyValue("environment/atmosphere/wind-speed-kt", 0.0);

    Atmosphere = FDMExec->GetAtmosphere();
    Winds = FDMExec->GetWinds();
    FCS = FDMExec->GetFCS();
    MassBalance = FDMExec->GetMassBalance();
    Propulsion = FDMExec->GetPropulsion();
    Aircraft = FDMExec->GetAircraft();
    Propagate = FDMExec->GetPropagate();
    Auxiliary = FDMExec->GetAuxiliary();
    Inertial = FDMExec->GetInertial();
    Aerodynamics = FDMExec->GetAerodynamics();
    GroundReactions = FDMExec->GetGroundReactions();
    Accelerations = FDMExec->GetAccelerations();

    bool result = false;

    if (!FDMExec->LoadModel(SGPath("aircraft"),
            SGPath(currentAircraft->enginesDir),
            SGPath(currentAircraft->systemsDir),
            currentAircraft->name))
    {
        cerr << "  JSBSim could not be started" << endl
             << endl;
        return false;
    }

    std::shared_ptr<JSBSim::FGInitialCondition> IC = FDMExec->GetIC();
    if (!IC->Load(SGPath("reset00.xml")))
    {
        cerr << "Initialization unsuccessful" << endl;
        return false;
    }

    il.SetLatitude(0.0);
    il.SetLongitude(0.0);

    m_controls.aileron = 0.0;
    m_controls.elevator = 0.0;

    reset(0.0);

    // PRINT SIMULATION CONFIGURATION
    FDMExec->PrintSimulationConfiguration();

    // Dump the simulation state (position, orientation, etc.)
    FDMExec->GetPropagate()->DumpState();

    // Perform trim if requested via the initialization file
    JSBSim::TrimMode icTrimRequested = (JSBSim::TrimMode)FDMExec->GetIC()->TrimRequested();
    if (icTrimRequested != JSBSim::TrimMode::tNone)
    {
        trimmer = new JSBSim::FGTrim(FDMExec, icTrimRequested);
        try
        {
            trimmer->DoTrim();

            if (FDMExec->GetDebugLevel() > 0)
                trimmer->Report();

            delete trimmer;
        }
        catch (string &msg)
        {
            cerr << endl
                 << msg << endl
                 << endl;
            return false;
        }
    }
    return true;
}

osg::Vec3f JSBSimPlugin::getOriginOffset() const
{
    auto geoDataPlugin = coVRPluginList::instance()->getPlugin("GeoData");
    if (geoDataPlugin)
    {
        return (static_cast<GeoDataLoader *>(geoDataPlugin))->rootOffset;
    }
    return osg::Vec3f();
}

void JSBSimPlugin::loadAvailableAircraft()
{
    auto configFile = config();

    auto aircraft = configFile->value<opencover::config::Section>("", "aircraft")->value();
    auto aircraftNames = aircraft.sections();

    for (auto sectionName : aircraftNames)
    {
        std::string aircraftName = sectionName;
        aircraftName.replace(0, 9, "");

        opencover::config::Section entry = configFile->value<opencover::config::Section>("aircraft", aircraftName)->value();

        // Check if the file exists and where exactly it is
        std::string geometryFile = entry.value<std::string>("", "geometry")->value();
        const char *geometryFileFound = coVRFileManager::instance()->getName(geometryFile.c_str());
        if (geometryFileFound == nullptr)
            geometryFileFound = "";

        // Read and apply the transform according to the aircraft config
        std::string transform = sectionName + ".geometryTransform";
        float x = configFloat(transform, "x", 0.0)->value();
        float y = configFloat(transform, "y", 0.0)->value();
        float z = configFloat(transform, "z", 0.0)->value();
        float h = configFloat(transform, "h", 0.0)->value();
        float p = configFloat(transform, "p", 0.0)->value();
        float r = configFloat(transform, "r", 0.0)->value();
        float s = configFloat(transform, "scale", 1000.0)->value();

        m_availableAircraft[aircraftName] = AircraftInfo {
            .name = aircraftName,
            .displayName = entry.value<std::string>("", "displayName", aircraftName)->value(),
            .geometryFile = geometryFileFound,
            .systemsDir = entry.value<std::string>("", "systemsDir", "systems")->value(),
            .enginesDir = entry.value<std::string>("", "enginesDir", "engine")->value(),
            .geometryTransform = osg::Matrix::scale(s, s, s)
                * osg::Matrix::rotate(
                    osg::DegreesToRadians(h), osg::Vec3(0, 0, 1),
                    osg::DegreesToRadians(p), osg::Vec3(1, 0, 0),
                    osg::DegreesToRadians(r), osg::Vec3(0, 1, 0))
                * osg::Matrix::translate(x, y, z),
        };

        if (m_defaultAircraft.empty())
        {
            m_defaultAircraft = aircraftName;
        }
    }
}

void JSBSimPlugin::readJoystickConfiguration()
{
    if (!coVRMSController::instance()->isMaster())
    {
        return;
    }

    joystickDev = (Joystick *)(Input::instance()->getDevice("joystick"));

    if (!joystickDev || joystickDev->numLocalJoysticks <= 0)
    {
        joystickDev = nullptr;
        return;
    }

    auto configFile = config();
    auto joystickSections = configFile->array<opencover::config::Section>("", "joysticks")->value();

    std::set<int> used_indexes;

    for (auto joystickSection : joystickSections)
    {
        std::string name = joystickSection.value<std::string>("", "name", "")->value();

        for (int i = 0; i < joystickDev->numLocalJoysticks; i++)
        {
            if (name.empty())
            {
                // Empty name means "catchall"; ignore joysticks already used
                // by some previous config.
                if (used_indexes.find(i) != used_indexes.end())
                {
                    continue;
                }
                else
                {
                    std::cout << "Found generic joystick '" << joystickDev->names[i] << "'" << std::endl
                              << "  with " << (int)joystickDev->number_axes[i] << " axes and " << (int)joystickDev->number_sliders[i] << " sliders." << std::endl;
                }
            }
            else if (joystickDev->names[i] != name)
            {
                continue;
            }

            for (auto section : joystickSection.array<opencover::config::Section>("", "actions")->value())
            {
                int axisNumber = section.value<int64_t>("", "axis", -1)->value();
                int sliderNumber = section.value<int64_t>("", "slider", -1)->value();
                std::string type = section.value<std::string>("", "type", "")->value();

                if (axisNumber >= 0 && joystickDev->number_axes[i] <= axisNumber)
                    continue;

                if (sliderNumber >= 0 && joystickDev->number_sliders[i] <= sliderNumber)
                    continue;

                if ((axisNumber >= 0) == (sliderNumber >= 0))
                    continue; // invalid config

                JoystickActionType actionType;
                if (type == "aileron")
                    actionType = AILERON;
                else if (type == "rudder")
                    actionType = RUDDER;
                else if (type == "elevator")
                    actionType = ELEVATOR;
                else if (type == "throttle")
                    actionType = THROTTLE;
                else if (type == "mixture")
                    actionType = MIXTURE;
                else
                    continue; // invalid config

                JoystickAction a;
                a.joystickNumber = i;
                a.type = actionType;
                a.axisNumber = axisNumber;
                a.sliderNumber = sliderNumber;
                a.invert = section.value<bool>("", "invert", false)->value();
                a.engine = (int)section.value<int64_t>("", "engine", -1)->value();
                m_joystickActions.push_back(a);

                used_indexes.emplace(i);
            }
        }
    }
}

bool JSBSimPlugin::init()
{
    delete udp;

    host = configString("Glider", "host", "141.58.8.212")->value();
    serverPort = configInt("Glider", "serverPort", 31319)->value();
    localPort = configInt("Glider", "localPort", 1234)->value();
    jsName = configString("JSBSim", "joystick", "Logitech X52 Professional H.O.T.A.S.")->value(); // SAITEK X-56 Joystick
    jsThrottleName = configString("JSBSim", "throttle", "SAITEK X-56 Throttle")->value();
    rudderName = configString("JSBSim", "rudder", "Thrustmaster T-Pendular-Rudder")->value();

    const char *rd = coVRFileManager::instance()->getName("share/covise/jsbsim");

    if (rd == nullptr)
        rd = "";
    RootDir = configString("JSBSim", "rootDir", rd)->value().c_str();

    // Initialize the coordinate system
    // TODO: move to a central geodata library
    coordTransformation = proj_create_crs_to_crs(PJ_DEFAULT_CTX, "EPSG:4326", "EPSG:25832", NULL);

    // Initialize menu
    JSBMenu = new ui::Menu("JSBSim", this);

    std::vector<std::string> aircraftNames;
    for (const auto &[key, _] : m_availableAircraft)
    {
        aircraftNames.push_back(key);
    }

    planeType = new ui::SelectionList(JSBMenu, "planeType");
    planeType->setList(aircraftNames);
    planeType->setCallback([this, aircraftNames](int val)
        { loadAircraft(aircraftNames[val]); });

    printCatalog = new ui::Action(JSBMenu, "printCatalog");
    printCatalog->setCallback([this]()
        {
        if (FDMExec)
            FDMExec->PrintPropertyCatalog(); });

    pauseButton = new ui::Button(JSBMenu, "pause");
    pauseButton->setState(false);
    pauseButton->setCallback([this](bool state)
        {
        if (FDMExec)
        {
            if (state)
                FDMExec->Hold();
            else
                FDMExec->Resume();
        } });

    debugButton = new ui::Button(JSBMenu, "debug");
    debugButton->setState(false);

    resetButton = new ui::Action(JSBMenu, "reset");
    resetButton->setCallback([this]()
        {
        initJSB();
        reset(); });

    upButton = new ui::Action(JSBMenu, "Up");
    upButton->setCallback([this]()
        {
        if (FDMExec)
            reset(100); });

    weatherGroup = new ui::Group(JSBMenu, "Weather");
    windLabel = new ui::Label(weatherGroup, "Wind");
    windLabel->setText("Wind");
    WX = new ui::EditField(weatherGroup, "X");
    WY = new ui::EditField(weatherGroup, "Y");
    WZ = new ui::EditField(weatherGroup, "Z");
    WX->setValue(0.0);
    WY->setValue(0.0);
    WZ->setValue(0.0);
    WX->setCallback(std::bind(&JSBSimPlugin::windChangedCallback, this));
    WY->setCallback(std::bind(&JSBSimPlugin::windChangedCallback, this));
    WZ->setCallback(std::bind(&JSBSimPlugin::windChangedCallback, this));

    labelVelocityX = new ui::Label(JSBMenu, "V_x");
    labelVelocityY = new ui::Label(JSBMenu, "V_y");
    labelVelocityZ = new ui::Label(JSBMenu, "V_z");

    bool ret = false;
    if (coVRMSController::instance()->isMaster())
    {
        std::string host = "";
        for (const auto &i : opencover::Input::instance()->discovery()->getDevices())
        {
            if (i->pluginName == "JSBSim")
            {
                host = i->address;
                if (i->deviceName == "GliderV2")
                {
                    deviceVersion = 2;
                }
                std::cerr << "JSBSim config: UDP: serverHost: " << host << ", localPort: " << localPort << ", serverPort: " << serverPort << std::endl;
                reset();
                udp = new UDPComm(host.c_str(), serverPort, localPort);
                if (!udp->isBad())
                {
                    ret = true;
                    // start();
                }
                else
                {
                    std::cerr << "Skateboard: falided to open local UDP port" << localPort << std::endl;
                    ret = false;
                }
                break;
            }
        }

        coVRMSController::instance()->sendSlaves(&ret, sizeof(ret));
    }
    else
    {
        coVRMSController::instance()->readMaster(&ret, sizeof(ret));
    }

    coVRNavigationManager::instance()->registerNavigationProvider(this);

    VrmlNamespace::addBuiltIn(VrmlNode::defineType<VrmlNodeThermal>());

    loadAircraft(m_defaultAircraft);

    return true;
}

void JSBSimPlugin::key(int type, int keySym, int mod)
{
    if (coVRMSController::instance()->isMaster())
    {
        if (type == osgGA::GUIEventAdapter::KEYDOWN)
        {
            switch (keySym)
            {
            case 'l':
            case 65363:
                m_controls.aileron += 0.1;
                break;

            case 'j':
            case 65361:
                m_controls.aileron -= 0.1;
                break;

            case 'm':
            case 65364:
                m_controls.elevator -= 0.1;
                break;

            case 'i':
            case 65362:
                m_controls.elevator += 0.1;
                break;

            case 'u':
                reset(100);
                break;

            case 'r':
                reset();
                break;

            case 'R':
                initJSB();
                reset();
                break;
            }

            m_controls.aileron = std::clamp(m_controls.aileron, -1.0, 1.0);
            m_controls.elevator = std::clamp(m_controls.elevator, -1.0, 1.0);
        }
    }
}

void JSBSimPlugin::updateInputs()
{
    // Read new values for all joystick actions
    for (auto &a : m_joystickActions)
    {
        a.update(joystickDev);
    }

    // Update aileron from the first changed joystick
    for (auto &a : m_joystickActions)
        if (a.type == AILERON && a.getChangedValue(m_controls.aileron))
            break;
    FCS->SetDaCmd(m_controls.aileron);

    // Update elevator from the first changed joystick
    for (auto &a : m_joystickActions)
        if (a.type == ELEVATOR && a.getChangedValue(m_controls.elevator))
            break;
    FCS->SetDeCmd(m_controls.elevator);

    // Update rudder from the first changed joystick
    for (auto &a : m_joystickActions)
        if (a.type == RUDDER && a.getChangedValue(m_controls.rudder))
            break;
    FCS->SetDrCmd(m_controls.rudder);

    // Update throttle and mixture, per engine
    for (unsigned int i = 0; i < Propulsion->GetNumEngines(); i++)
    {
        double value;
        for (auto &a : m_joystickActions)
        {
            if (a.type == THROTTLE && (a.engine == -1 || a.engine == i) && a.getChangedValue(value))
            {
                FCS->SetThrottleCmd(i, value * 0.5 + 0.5);
                break;
            }
        }

        for (auto &a : m_joystickActions)
        {
            if (a.type == MIXTURE && (a.engine == -1 || a.engine == i) && a.getChangedValue(value))
            {
                FCS->SetMixtureCmd(i, value * 0.5 + 0.5);
                break;
            }
        }
    }

    // Make sure all engines are running (incl. starter/magnetos)
    for (unsigned int i = 0; i < Propulsion->GetNumEngines(); i++)
    {
        auto engine = Propulsion->GetEngine(i);
        engine->SetStarter(1);
        engine->SetRunning(1);
        if (engine->GetType() == JSBSim::FGEngine::etPiston)
            std::static_pointer_cast<JSBSim::FGPiston>(engine)->SetMagnetos(3); // mode 3 = both left and right running
    }
}

bool JSBSimPlugin::update()
{
    // Receive input data even when disabled or paused, so we don't fill buffers
    updateUdp();

    if (isEnabled())
    {
        bool stopNav = false;
        if (coVRMSController::instance()->isMaster())
        {
            bool result = true;
            if (VRSceneGraph::instance()->getTransform()->getMatrix() != lastPos) // if someone else moved e.g. a new viewpoint has been set, then do a reset to this position
            {
                lastPos = VRSceneGraph::instance()->getTransform()->getMatrix();
                reset();
            }
            else
            {
                updateInputs();

                while (FDMExec->GetSimTime() + SimStartTime < cover->frameTime())
                {
                    if (FDMExec->Holding())
                        break;
                    try
                    {
                        result = FDMExec->Run();
                        if (!result)
                        {
                            stopNav = true;
                            break;
                        }
                    }
                    catch (std::string s)
                    {
                        fprintf(stderr, "oops, exception %s\n", s.c_str());
                        stopNav = true;
                    }
                    catch (...)
                    {
                        fprintf(stderr, "oops, exception\n");
                        stopNav = true;
                    }
                }
                if (result)
                {
                    if (debugButton->state())
                    {
                        FDMExec->GetPropagate()->DumpState();
                    }

                    auto vehicleState = Propagate->GetVState();
                    auto location = vehicleState.vLocation;

                    PJ_COORD c;
                    c.lpz.lam = location.GetLatitudeDeg();
                    c.lpz.phi = location.GetLongitudeDeg();
                    c.lpz.z = location.GetGeodAltitude() * METERS_PER_FOOT;
                    PJ_COORD c_out = proj_trans(coordTransformation, PJ_FWD, c);

                    float scale = cover->getScale();
                    osg::Vec3 position(c_out.enu.e, c_out.enu.n, c_out.enu.u);
                    position += getOriginOffset();
                    auto trans = osg::Matrix::translate(position * scale);

                    float heading = Propagate->GetEuler(JSBSim::FGJSBBase::ePsi);
                    float roll = Propagate->GetEuler(JSBSim::FGJSBBase::ePhi);
                    float pitch = Propagate->GetEuler(JSBSim::FGJSBBase::eTht);

                    // std::cout << " Heading " << heading << " Roll " << roll << " Pitch " << pitch << std::endl;
                    // auto rot = osg::Matrix::rotate(
                    //     -pitch, osg::Vec3(1, 0, 0),
                    //     roll, osg::Vec3(0, 1, 0),
                    //     heading, osg::Vec3(0, 0, 1));

                    auto rot = osg::Matrix::rotate(roll, osg::Vec3(0, 1, 0)) * osg::Matrix::rotate(pitch, osg::Vec3(1, 0, 0)) * osg::Matrix::rotate(-heading, osg::Vec3(0, 0, 1));

                    osg::Matrix newPos = osg::Matrix::inverse(/*osg::Matrix::rotate(-M_PI_2, 0, 0, 1.0) * eyePoint */ rot * trans);

                    if (newPos.isNaN())
                    {
                        stopNav = true;
                    }
                    else if (FDMExec && !FDMExec->Holding())
                    {
                        lastPos = newPos;

                        // Update wind speed gradually
                        m_windVelocityCurrent += (m_windVelocityTarget - m_windVelocityCurrent) * 0.1f * cover->frameDuration();

                        float speedX = vehicleState.vUVW(1);
                        float speedY = vehicleState.vUVW(2);
                        float speedZ = vehicleState.vUVW(3);

                        labelVelocityX->setText("V_x: " + std::to_string(speedX));
                        labelVelocityY->setText("V_y: " + std::to_string(speedY));
                        labelVelocityZ->setText("V_z: " + std::to_string(speedZ));

                        if (varioSource)
                        {
                            float varioPitch = (std::clamp(-speedZ / 10.f, -1.f, 1.f) + 1.f) * 0.3f + 0.8f;
                            varioSource->setPitch(varioPitch);

                            float varioVolume = std::clamp((fabs((varioPitch - 1.f)) * 5.f) - 0.2f, 0.f, 1.f);
                            varioSource->setIntensity(varioVolume);
                        }

                        if (windSource)
                        {
                            windSource->setIntensity(1.0);
                        }

                        if (engineSource)
                        {
                            engineSource->setIntensity(1.0);
                            engineSource->setPitch(1.0);
                        }

                        // Update winds
                        Winds->SetWindNED(m_windVelocityCurrent.y(), m_windVelocityCurrent.x(), -m_windVelocityCurrent.z());
                        Winds->SetTurbGain(m_windTurbulenceCurrent);

                        // Reset target velocity and turbulence so addThermal() can add to it each frame
                        m_windVelocityTarget = m_windVelocitySetting;
                        m_windTurbulenceTarget = 0;
                    }
                }
            }

            coVRMSController::instance()->sendSlaves((char *)&stopNav, sizeof(stopNav));
            if (!stopNav)
            {
                coVRMSController::instance()->sendSlaves((char *)lastPos.ptr(), sizeof(lastPos));
                VRSceneGraph::instance()->getTransform()->setMatrix(lastPos);
                coVRCollaboration::instance()->SyncXform();
            }
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)&stopNav, sizeof(stopNav));
            if (!stopNav)
            {
                coVRMSController::instance()->readMaster((char *)lastPos.ptr(), sizeof(lastPos));
                VRSceneGraph::instance()->getTransform()->setMatrix(lastPos);
                coVRCollaboration::instance()->SyncXform();
            }
        }

        if (stopNav)
        {
            coVRNavigationManager::instance()->setNavMode(coVRNavigationManager::Walk);
            return false;
        }
        return true;
    }
    return false;
}

void JSBSimPlugin::setEnabled(bool flag)
{
    coVRNavigationProvider::setEnabled(flag);

    if (flag)
    {
        cover->getScene()->addChild(geometryTrans.get());
    }
    else
    {
        cover->getScene()->removeChild(geometryTrans.get());
    }
    if (coVRMSController::instance()->isMaster())
    {
        if (flag)
        {
            reset();
            if (varioSource)
                varioSource->play();
            if (windSource)
                windSource->play();
            if (engineSource)
                engineSource->play();
        }
        else
        {
            if (varioSource)
                varioSource->stop();
            if (windSource)
                windSource->stop();
            if (engineSource)
                engineSource->stop();
        }
        if (udp)
        {
            if (flag)
            {
                udp->send("start");
            }
            else
            {
                udp->send("stop");
            }
        }
    }
}

void JSBSimPlugin::updateUdp()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);

    if (!udp)
        return;

    char buffer[24];
    int received_bytes = udp->receive(buffer, 24, 0.0);

    // Error response
    if (received_bytes < 0)
    {
        static bool firstTime = true;
        if (firstTime)
        {
            std::cerr << "JSBSimPlugin::updateUdp: error while reading data" << std::endl;
            firstTime = false;
        }
        return;
    }

    // Special case, if the message is exactly  the size
    if (received_bytes == sizeof(GliderValues))
    {
        GliderValues gliderValues;
        memcpy(&gliderValues, buffer, sizeof(GliderValues));

        float leftLine = std::clamp(gliderValues.left / 1280.0, 0.0, 1.0);
        float rightLine = std::clamp(gliderValues.right / 1280.0, 0.0, 1.0);
        float angleValue = -(gliderValues.angle / 180.0);

        double elevatorMin = -1, elevatorMax = 1, aileronMax = 1;
        m_controls.elevator = -((leftLine + rightLine) / 2 * (elevatorMax - elevatorMin) + elevatorMin);
        m_controls.aileron = (rightLine - leftLine + angleValue) * aileronMax;
        m_controls.throttle = 0.0;
        m_controls.mixture = 0.0;

        std::cerr << "JSBSimPlugin::left:" << gliderValues.left << "     " << leftLine << "     " << gliderValues.angle << std::endl;
        std::cerr << "JSBSimPlugin::right:" << gliderValues.right << "     " << rightLine << "     " << angleValue << std::endl;
        return;
    }

    // Special case for old format without throttle
    if (received_bytes == 16)
    {
        memcpy(&m_controls, buffer, 16);
        byteSwap(m_controls.aileron);
        byteSwap(m_controls.elevator);
        m_controls.throttle = 0.0;
        m_controls.mixture = 0.0;
    }
    else if (received_bytes == 24)
    {
        memcpy(&m_controls, buffer, 24);
        byteSwap(m_controls.aileron);
        byteSwap(m_controls.elevator);
        byteSwap(m_controls.throttle);
        m_controls.mixture = 0.0;
    }
}

void JSBSimPlugin::addThermal(const osg::Vec3 &velocity, float turbulence)
{
    m_windVelocityTarget += velocity;
    m_windTurbulenceTarget += turbulence;
}

COVERPLUGIN(JSBSimPlugin)
