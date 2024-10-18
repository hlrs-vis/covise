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
#include <cover/input/input.h>
#include "cover/input/deviceDiscovery.h"
#include <util/UDPComm.h>
#include <util/byteswap.h>
#include <util/unixcompat.h>
#include <stdlib.h>

#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <cover/coVRFileManager.h>
#include <osg/Vec3>

#include <cover/input/input.h>

JSBSimPlugin* JSBSimPlugin::plugin = NULL;

JSBSimPlugin::JSBSimPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("JSBSimPlugin", cover->ui)
, coVRNavigationProvider("JSBsim", this)
{
    fprintf(stderr, "JSBSimPlugin::JSBSimPlugin\n");
    geometryTrans = new osg::MatrixTransform();

    if (coVRMSController::instance()->isMaster())
    {

        remoteSoundServer = configString("Sound", "server", "localhost")->value();
        remoteSoundPort = configInt("Sound", "port", 31805)->value();

        const char* VS = coVRFileManager::instance()->getName("share/covise/jsbsim/Sounds/vario.wav");
        if (VS == nullptr)
            VS = "";
        VarioSound = configString("Sound", "vario", VS)->value();
        const char* WS = coVRFileManager::instance()->getName("share/covise/jsbsim/Sounds/wind1.wav");
        if (WS == nullptr)
            WS = "";
        WindSound = configString("Sound", "wind", WS)->value();
#if defined(_MSC_VER)
        // _clearfp();
        // _controlfp(_controlfp(0, 0) & ~(_EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW),
        //     _MCW_EM);
#elif defined(__GNUC__) && !defined(sgi) && !defined(__APPLE__)
        //feenableexcept(FE_DIVBYZERO | FE_INVALID);
#endif

        rsClient = new remoteSound::Client(remoteSoundServer, remoteSoundPort, "JSBSim");
        varioSound = rsClient->getSound(VarioSound);
        windSound = rsClient->getSound(WindSound);
        varioSound->setLoop(true, -1);
        windSound->setLoop(true, -1);
    }
    plugin = this;
    udp = 0;
}
void JSBSimPlugin::updateTrans()
{
    float tx = cX->value();
    float ty = cY->value();
    float tz = cZ->value();
    float th = cH->value() / 180 * M_PI;
    float tp = cP->value() / 180 * M_PI;
    float tr = cR->value() / 180 * M_PI;
    float ts = cS->value();
    osg::Matrix gt = osg::Matrix::scale(ts, ts, ts) * osg::Matrix::rotate(th, osg::Vec3(0, 0, 1), tp, osg::Vec3(1, 0, 0), tr, osg::Vec3(0, 1, 0)) * osg::Matrix::translate(tx, ty, tz);
    geometryTrans->setMatrix(gt);
}

void JSBSimPlugin::initAircraft()
{
    std::string AircraftGeometry = configString(currentAircraft, "geometry", "share/covise/jsbsim/geometry/paraglider.osgb")->value();

    const char* GF = coVRFileManager::instance()->getName(AircraftGeometry.c_str());
    if (GF == nullptr)
        GF = "";
    geometryFile = GF;
    cX = configFloat(currentAircraft, "x", 0.0);
    cY = configFloat(currentAircraft, "y", 0.0);
    cZ = configFloat(currentAircraft, "z", 0.0);
    cH = configFloat(currentAircraft, "h", 0.0);
    cP = configFloat(currentAircraft, "p", 0.0);
    cR = configFloat(currentAircraft, "r", 0.0);
    cS = configFloat(currentAircraft, "scale", 1000.0);

    tX->setValue(cX->value());
    tY->setValue(cY->value());
    tZ->setValue(cZ->value());
    tH->setValue(cH->value() / 180 * M_PI);
    tP->setValue(cP->value() / 180 * M_PI);
    tR->setValue(cR->value() / 180 * M_PI);
    tS->setValue(cS->value());

    updateTrans();
    while (geometryTrans->getNumChildren())
        geometryTrans->removeChild(0,1); //remove old geometries
    osg::Node* n = osgDB::readNodeFile(geometryFile.c_str());

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

//! reimplement to do early cleanup work and return false to prevent unloading
bool JSBSimPlugin::destroy()
{
    if (VrmlNodeThermal::numThermalNodes > 0)
    {
        cerr << "Thermal nodes are still in use, can't delete JSBSimPlugin" << endl;
        return false;
    }
    
    return true;
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
    delete DebugButton;
    delete resetButton;
    delete upButton;
    delete JSBMenu;
    delete rsClient;
}


}

void JSBSimPlugin::reset(double dz)
{
    if(FDMExec==nullptr)
    {
        initJSB();
    }
    FDMExec->Setdt(1.0 / 120.0);
    frame_duration = FDMExec->GetDeltaT();

    FDMExec->Setsim_time(0.0);
    SimStartTime = cover->frameTime();
    osg::Vec3 viewerPosInFeet = cover->getInvBaseMat().getTrans() / 0.3048;

    //viewerPosInFeet.set(0, 0, 0);
    osg::Vec3 dir;
    //dir.set(0, 1, 0);

    double radius = (Propagate->GetVState().vLocation.GetSeaLevelRadius() + viewerPosInFeet[2]);
    double ecX, ecY, ecZ;
    ecX = viewerPosInFeet[2] + radius;
    ecY = -viewerPosInFeet[1];
    ecZ = viewerPosInFeet[0];
    double r02 = ecX * ecX + ecY * ecY;
    double rxy = sqrt(r02);
    double mLon, mLat;
    eyePoint.makeTranslate(Aircraft->GetXYZep(1) * 25.4, Aircraft->GetXYZep(2) * 25.4, Aircraft->GetXYZep(3) * 25.4);


    // Compute the longitude and latitude itself
    if (ecX == 0.0 && ecY == 0.0)
        mLon = 0.0;
    else
        mLon = atan2(ecY, ecX);

    if (rxy == 0.0 && ecZ == 0.0)
        mLat = 0.0;
    else
        mLat = atan2(ecZ, rxy);
    osg::Matrix viewer = cover->getBaseMat();
    for (int i = 0; i < 3; i++)
        dir[i] = viewer(1, i);
    osg::Vec3 y(0, 1, 0);
    dir.normalize();
    /*

        double v[3];

        v[0] = viewerPosInFeet[0] - projectOffset[0];
        v[1] = viewerPosInFeet[1] - projectOffset[1];
        v[2] = viewerPosInFeet[2] - projectOffset[2];
        int error = pj_transform(pj_to, pj_from, 1, 0, v,v+1, v+2);
        if (error != 0)
        {
            fprintf(stderr, "%s \n ------ \n", pj_strerrno(error));
        }
        mLon = v[0];
        mLat = v[1];*/


    std::shared_ptr<JSBSim::FGInitialCondition> IC = FDMExec->GetIC();
    if (!IC->Load(SGPath(resetFile))) {
        cerr << "Initialization unsuccessful" << endl;
    }
    IC->SetLatitudeRadIC(mLat);
    IC->SetLongitudeRadIC(mLon);
    IC->SetAltitudeASLFtIC(viewerPosInFeet[2] - Aircraft->GetXYZep(3) * 0.0833 + (dz) / 0.3048);
    IC->SetAltitudeAGLFtIC(viewerPosInFeet[2] - Aircraft->GetXYZep(3) * 0.0833 + (dz) / 0.3048);
    IC->SetPsiRadIC((-(atan2(y[1], y[0]) - atan2(dir[1], dir[0]))) - M_PI_2);// heading
    //IC->SetPsiRadIC(0.0); // heading
    //IC->SetThetaRadIC(-20.0*DEG_TO_RAD); // pitch from reset file beause the paraglider is sensitive to start pitch
    IC->SetPhiRadIC(0.0); // roll
    //IC->SetUBodyFpsIC(10); // U is Body X direction --> forwards



    Propagate->InitModel();
    Propagate->InitializeDerivatives();
    FDMExec->RunIC();

    Winds->SetWindNED(WY->number(), WX->number(), -WZ->number());
    targetVelocity.set(WX->number(), WY->number(), WZ->number());
    currentVelocity.set(WX->number(), WY->number(), WZ->number());
    JSBSim::FGPropagate::VehicleState location = Propagate->GetVState();

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
    FDMExec->GetPropertyManager()->Tie("simulation/frame_start_time", &actual_elapsed_time);
    FDMExec->GetPropertyManager()->Tie("simulation/cycle_duration", &cycle_duration);


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

    Winds->SetWindNED(WY->number(), WX->number(), -WZ->number());

    if (nohighlight) FDMExec->disableHighLighting();

    bool result = false;

    std::string line = configString(currentAircraft, "scriptName", "")->value();
    ScriptName.set(line);
    
    AircraftDir = configString(currentAircraft, "aircraftDir", "aircraft")->value();
    EnginesDir = configString(currentAircraft, "enginesDir", "engine")->value();
    SystemsDir = configString(currentAircraft, "systemsDir", "paraglider/Systems")->value();
    resetFile = configString(currentAircraft, "resetFile", "reset00.xml")->value();

    // *** OPTION A: LOAD A SCRIPT, WHICH LOADS EVERYTHING ELSE *** //
    if (!ScriptName.isNull()) {

        result = FDMExec->LoadScript(ScriptName, 0.008333, SGPath(resetFile));

        if (!result) {
            cerr << "Script file " << ScriptName << " was not successfully loaded" << endl;
            return false;
        }

        // *** OPTION B: LOAD AN AIRCRAFT AND A SET OF INITIAL CONDITIONS *** //
    }
    else if (!currentAircraft.empty() || !resetFile.length() == 0) {

        if (catalog) FDMExec->SetDebugLevel(0);

        if (!FDMExec->LoadModel(SGPath(AircraftDir),
            SGPath(EnginesDir),
            SGPath(SystemsDir),
            currentAircraft)) {
            cerr << "  JSBSim could not be started" << endl << endl;
            return false;
        }

    }
    else {
        cout << "  No Aircraft, Script, or Reset information given" << endl << endl;
        return false;
    }
    std::shared_ptr<JSBSim::FGInitialCondition> IC = FDMExec->GetIC();
    if (!IC->Load(SGPath(resetFile))) {
        cerr << "Initialization unsuccessful" << endl;
        return false;
    }

    il.SetLatitude(0.0);
    il.SetLongitude(0.0);

    fgcontrol.aileron = 0.0;
    fgcontrol.elevator = 0.0;
    gliderValues.left = 0.0;
    gliderValues.right = 0.0;
    gliderValues.angle = 0.0;
    gliderValues.speed = 0.0;
    gliderValues.state = 0;

    reset(0.0);

    // PRINT SIMULATION CONFIGURATION
    FDMExec->PrintSimulationConfiguration();

    // Dump the simulation state (position, orientation, etc.)
    FDMExec->GetPropagate()->DumpState();

    // Perform trim if requested via the initialization file
    JSBSim::TrimMode icTrimRequested = (JSBSim::TrimMode)FDMExec->GetIC()->TrimRequested();
    if (icTrimRequested != JSBSim::TrimMode::tNone) {
        trimmer = new JSBSim::FGTrim(FDMExec, icTrimRequested);
        try {
            trimmer->DoTrim();

            if (FDMExec->GetDebugLevel() > 0)
                trimmer->Report();

            delete trimmer;
        }
        catch (string& msg) {
            cerr << endl << msg << endl << endl;
            return false;
        }
    }
    return true;
}

bool JSBSimPlugin::init()
{
    delete udp;
    
    host = configString("Glider", "host", "141.58.8.212")->value();
    serverPort = configInt("Glider", "serverPort", 31319)->value();
    localPort = configInt("Glider", "localPort", 1234)->value();

    const char* rd = coVRFileManager::instance()->getName("share/covise/jsbsim");
   
    if(rd==nullptr)
        rd="";
    RootDir = configString("JSBSim", "rootDir", rd)->value().c_str();



    //mapping of coordinates
#ifdef WIN32
    const char* pValue="";
    size_t len;
    char* ncpValue = (char*)pValue;
    errno_t err = _dupenv_s(&ncpValue, &len, "COVISEDIR");
    if (err)
        pValue = "";
#else
    const char* pValue = getenv("COVISEDIR");
    if (!pValue)
        pValue = "";
#endif
    coviseSharedDir = std::string(pValue) + "/share/covise/";

    //std::string proj_from = configString("Projection", "from", "+proj=latlong +datum=WGS84")->value();
    //if (!(pj_from = pj_init_plus(proj_from.c_str())))
    //{
    //    fprintf(stderr, "ERROR: pj_init_plus failed with pj_from = %s\n", proj_from.c_str());
    //}

    //std::string proj_to = configString("Projection", "to", "+proj=tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=9703.397 +y_0=-5384244.453 +ellps=bessel +datum=potsdam")->value();// +nadgrids=" + dir + std::string("BETA2007.gsb");

    //if (!(pj_to = pj_init_plus(proj_to.c_str())))
    //{
    //    fprintf(stderr, "ERROR: pj_init_plus failed with pj_to = %s\n", proj_to.c_str());
    //}

    coordTransformation = proj_create_crs_to_crs(PJ_DEFAULT_CTX, 
        "+proj=latlong +datum=WGS84", 
        "+proj=tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=9703.397 +y_0=-5384244.453 +ellps=bessel +datum=potsdam",
        NULL);

    projectOffset[0] = configFloat("Projection", "offsetX", 0)->value();
    projectOffset[1] = configFloat("Projection", "offsetY", 0)->value();
    projectOffset[2] = configFloat("Projection", "offsetZ", 0)->value();

    JSBMenu = new ui::Menu("JSBSim", this);

    aircrafts = configStringArray("JSBSim", "aircrafts", { "Paraglider", "c172b", "J3Cub" });
    currentAircraft = aircrafts->value()[0]; 
    planeType = new ui::SelectionList(JSBMenu, "planeType");
    planeType->setList(aircrafts->value());
    planeType->setCallback([this](int val) {
        currentAircraft = aircrafts->value()[val];
        initAircraft();
        });

    printCatalog = new ui::Action(JSBMenu, "printCatalog");
    printCatalog->setCallback([this]() {
        if (FDMExec)
            FDMExec->PrintPropertyCatalog();
        });
    pauseButton = new ui::Button(JSBMenu, "pause");
    pauseButton->setState(false);
    pauseButton->setCallback([this](bool state) {
        if (FDMExec)
        {
            if (state)
                FDMExec->Hold();
            else
                FDMExec->Resume();
        }
        });

    DebugButton = new ui::Button(JSBMenu, "debug");
    DebugButton->setState(false);

    resetButton = new ui::Action(JSBMenu, "reset");
    resetButton->setCallback([this]() {
        initJSB();
        reset();
        });

    upButton = new ui::Action(JSBMenu, "Up");
    upButton->setCallback([this]() {
        if (FDMExec)
            reset(100);
        });



    Weather = new ui::Group(JSBMenu, "Weather");
    Geometry = new ui::Group(JSBMenu, "Geometry");
    WindLabel = new ui::Label(Weather, "Wind");
    WX = new ui::EditField(Weather, "X");
    WY = new ui::EditField(Weather, "Y");
    WZ = new ui::EditField(Weather, "Z");
    WX->setValue(0.0);
    WY->setValue(0.0);
    WZ->setValue(0.0);
    WX->setCallback([this](std::string v) {
        if(Winds) Winds->SetWindNED(WY->number(), WX->number(), -WZ->number());
        });
    WY->setCallback([this](std::string v) {
        if(Winds) Winds->SetWindNED(WY->number(), WX->number(), -WZ->number());
        });
    WZ->setCallback([this](std::string v) {
        if(Winds) Winds->SetWindNED(WY->number(), WX->number(), -WZ->number());
        });
    tX = new ui::EditField(Geometry, "X");
    tY = new ui::EditField(Geometry, "Y");
    tZ = new ui::EditField(Geometry, "Z");
    tX->setValue(0.0);
    tY->setValue(0.0);
    tZ->setValue(0.0);
    tX->setCallback([this](std::string v) {
        if (cX)
        {
            *cX = double(tX->number());
            updateTrans();
        }
        });
    tY->setCallback([this](std::string v) {
        if (cY)
        {
            *cY = double(tY->number());
        updateTrans();
        }
        });
    tZ->setCallback([this](std::string v) {
        if (cZ)
        {
            *cZ = double(tZ->number());
        updateTrans();
        }
        });
    tH = new ui::EditField(Geometry, "H");
    tP = new ui::EditField(Geometry, "P");
    tR = new ui::EditField(Geometry, "R");
    tH->setValue(0.0);
    tP->setValue(0.0);
    tR->setValue(0.0);
    tH->setCallback([this](std::string v) {
        if (cH)
        {
            *cH = double(tH->number() / M_PI * 180);
        updateTrans();
        }
        });
    tP->setCallback([this](std::string v) {
        if (cP)
        {
            *cP = double(tP->number() / M_PI * 180);
        updateTrans();
        }
        });
    tR->setCallback([this](std::string v) {
        if (cR)
        {
            *cR = double(tR->number() / M_PI * 180);
        updateTrans();
        }
        });
    tS = new ui::EditField(Geometry, "S");
    tS->setValue(1000.0);
    tS->setCallback([this](std::string v) {
        if (cS)
        {
            *cS = double(tS->number());
            updateTrans();
        }
        });
    VLabel = new ui::Label(JSBMenu,"V");
    VzLabel = new ui::Label(JSBMenu,"Vz");

    currentVelocity.set(WX->number(), WY->number(), WZ->number());
    currentTurbulence = 0;

        bool ret = false;
        if (coVRMSController::instance()->isMaster())
        {
            std::string host = "";
            for (const auto& i : opencover::Input::instance()->discovery()->getDevices())
            {
                if (i->pluginName == "JSBSim")
                {
                    host = i->address;
		    if(i->deviceName == "GliderV2")
		    {
		        deviceVersion = 2;
		    }
                    std::cerr << "JSBSim config: UDP: serverHost: " << host << ", localPort: " << localPort << ", serverPort: " << serverPort << std::endl;
                    reset();
                    udp = new UDPComm(host.c_str(), serverPort, localPort);
                    if (!udp->isBad())
                    {
                        ret = true;
                        //start();
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

    VrmlNamespace::addBuiltIn(VrmlNodeTemplate::defineType<VrmlNodeThermal>());

    initAircraft();
    return true;
}

void JSBSimPlugin::key(int type, int keySym, int mod)
{
if (coVRMSController::instance()->isMaster())
        {
    (void)mod;
    switch (type)
    {
    case (osgGA::GUIEventAdapter::KEYDOWN):
        if (keySym == 'l' || keySym == 65363)
        {
            if (fgcontrol.aileron < 0.99)
                fgcontrol.aileron += 0.1;
        }
        else if (keySym == 'j' || keySym == 65361)
        {
            if (fgcontrol.aileron > -0.99)
                fgcontrol.aileron -= 0.1;
        }
        else if (keySym == 'm' || keySym == 65364)
        {
            if (fgcontrol.elevator < 0.99)
                fgcontrol.elevator += 0.1;
        }
        else if (keySym == 'i' || keySym == 65362)
        {
            if (fgcontrol.elevator > -0.99)
                fgcontrol.elevator -= 0.1;
        }
        else if (keySym == 'u')
        {
            reset(100);
        }
        else if (keySym == 'r')
        {
            reset();
        }
        else if (keySym == 'R')
        {
            initJSB();
            reset();
        }
        break;
    }
    fprintf(stderr, "Keysym: %d\n", keySym);
    fprintf(stderr, "Aileron: %lf Elevator %lf\n", fgcontrol.aileron, fgcontrol.elevator);
}
}

bool
JSBSimPlugin::update()
{
    joystickDev = (Joystick*)(Input::instance()->getDevice("joystick"));
    if (joystickDev->numLocalJoysticks > 0)
    {
        for (int i = 0; i < joystickDev->numLocalJoysticks; i++)
        {
            if (joystickDev->number_axes[i] == 6 && joystickDev->number_sliders[i] == 1)
            {
                Joysticknumber = i;
            }
            if (joystickDev->number_axes[i] == 3 && joystickDev->number_sliders[i] == 0)
            {
                Ruddernumber = i;
            }
        }

    }
    else
    {
        joystickDev = nullptr;
    }

    if (joystickDev != nullptr)
    {/*
        std::cout << */
            //"\n[0][0]:" << joystickDev->buttons[0][0] <<
            //", [0][1]:" << joystickDev->buttons[0][1] <<
            //", [0][2]:" << joystickDev->buttons[0][2] <<
            //", [0][3]:" << joystickDev->buttons[0][3] <<
            //", [0][4]:" << joystickDev->buttons[0][4] <<
            //", [0][5]:" << joystickDev->buttons[0][5] <<
            //", [0][6]:" << joystickDev->buttons[0][6] <<
            //", [0][7]:" << joystickDev->buttons[0][7] <<
      
            //"\nsliders[0]:" << joystickDev->sliders[0][0] <<
            //"axes[0]:" << joystickDev->axes[0][0] << 
            //", axes[1]:" << joystickDev->axes[0][1] << std::endl;
    }

    if (joystickDev) {

        // Read joystick axis values
        float joystickX = joystickDev->axes[Joysticknumber][0];
        float joystickY = joystickDev->axes[Joysticknumber][1];
        //float throttle = joystickDev->sliders[0][0];
        //std::cout << "joystickX = " << joystickX << ", joystickY = " << joystickY << ", throttle = " << throttle;

        // Map joystick input to control surfaces
        fgcontrol.aileron = joystickX;  //joystickDev->axes[0][0];
        fgcontrol.elevator = -joystickY;  //joystickDev->axes[0][1];
        //gliderValues.speed = throttle;
        //gliderValues.state = joystickDev->buttons[0][0];
    }

    //gliderValues.left = joystickX;
    //gliderValues.right = joystickY;
    //gliderValues.speed = throttle;


if (coVRMSController::instance()->isMaster())
        {
    updateUdp();
    //std::cout << "Entered coVRMSController::instance()_>isMAster()" << std::endl;
    rsClient->update();
}

    if (isEnabled())
    {

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
            FCS->SetDaCmd(fgcontrol.aileron);
            std::cout << "\nfgcontrol.aileron:" << fgcontrol.aileron << std::endl;
            FCS->SetDeCmd(fgcontrol.elevator);
            std::cout << "fgcontrol.elevator:" << fgcontrol.elevator << std::endl;
            if (joystickDev)
            {
                if(Ruddernumber>=0)
                    FCS->SetDrCmd(joystickDev->axes[Ruddernumber][2]);
                else if(Joysticknumber>=0)
                    FCS->SetDrCmd(-joystickDev->axes[Joysticknumber][5]);
            }

            for (unsigned int i = 0; i < Propulsion->GetNumEngines(); i++) {
                if (joystickDev)
                {
                    FCS->SetThrottleCmd(i,1.0-((1+joystickDev->axes[Joysticknumber][2])/2.0));
                    FCS->SetMixtureCmd(i, joystickDev->sliders[Joysticknumber][0]);
                }
                else
                {
                    FCS->SetThrottleCmd(i, 1.0);
                    FCS->SetMixtureCmd(i, 1.0);
                }

                switch (Propulsion->GetEngine(i)->GetType())
                {
                case JSBSim::FGEngine::etPiston:
                { // FGPiston code block
                    auto piston_engine = static_pointer_cast<JSBSim::FGPiston>(Propulsion->GetEngine(i));
                    piston_engine->SetMagnetos(3);
                    break;
                } // end FGPiston code block
                }
                { // FGEngine code block
                    auto eng = Propulsion->GetEngine(i);
                    eng->SetStarter(1);
                    eng->SetRunning(1);
                } // end FGEngine code block
            }

            while (FDMExec->GetSimTime() + SimStartTime < cover->frameTime())
            {
                if (FDMExec->Holding()) break;
                try
                {
                    result = FDMExec->Run();
                    if (!result)
                    {
                        coVRNavigationManager::instance()->setNavMode(coVRNavigationManager::Walk);
                        break;
                    }
                }
                catch (std::string s)
                {
                    fprintf(stderr, "oops, exception %s\n", s.c_str());
                    coVRNavigationManager::instance()->setNavMode(coVRNavigationManager::Walk);
                }
                catch (...)
                {
                    fprintf(stderr, "oops, exception\n");
                    coVRNavigationManager::instance()->setNavMode(coVRNavigationManager::Walk);
                }
            }
            if (result)
            {
                if (DebugButton->state())
                {
                    FDMExec->GetPropagate()->DumpState();
                }

                osg::Matrix rot;
                rot.makeRotate(Propagate->GetEuler(JSBSim::FGJSBBase::eTht), osg::Vec3(0, -1, 0), Propagate->GetEuler(JSBSim::FGJSBBase::ePhi), osg::Vec3(1, 0, 0), Propagate->GetEuler(JSBSim::FGJSBBase::ePsi), osg::Vec3(0, 0, -1));
                osg::Matrix trans;

                JSBSim::FGPropagate::VehicleState location = Propagate->GetVState();
                float scale = cover->getScale();

                /*
                double v[3];

                v[0] = location.vLocation.GetLongitude();
                v[1] = location.vLocation.GetLatitude();
                v[2] = location.vLocation.GetGeodAltitude() * 0.3048;
                int error = pj_transform(pj_from, pj_to, 1, 0, v, v + 1, v + 2);
                if (error != 0)
                {
                    fprintf(stderr, "%s \n ------ \n", pj_strerrno(error));
                }

                trans.makeTranslate(v[1]*scale, v[0] * scale, v[2] * scale);

                */

                trans.makeTranslate(location.vLocation(3) * 0.3048 * scale, -location.vLocation(2) * 0.3048 * scale, (location.vLocation(1) - location.vLocation.GetSeaLevelRadius()) * 0.3048 * scale);
                osg::Matrix preRot;
                preRot.makeRotate(-M_PI_2, 0, 0, 1.0);
                osg::Matrix newPos = osg::Matrix::inverse(preRot * eyePoint * rot * trans);
                /*JSBSim::FGMatrix33 tb2lMat = Propagate->GetTb2l();
                Plane.makeIdentity();
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        Plane(i, j) = tb2lMat.Entry(i+1, j+1);
                 Plane.invert(Plane); */

                if (FDMExec && !FDMExec->Holding() && !newPos.isNaN())
                {
                    lastPos = newPos;
                    if (targetVelocity != currentVelocity)
                    {
                        osg::Vec3 diff = targetVelocity - currentVelocity;
                        if (diff.length2() < 0.0001)
                        {
                            currentVelocity = targetVelocity;
                        }
                        else
                        {
                            currentVelocity += diff * 0.1;
                        }
                    }
                    float vSpeed = location.vUVW(3);
                    VzLabel->setText("Vz: " + std::to_string(vSpeed));
                    VLabel->setText("V: " + std::to_string(location.vUVW.Magnitude(1,2)));
                    float pitch = -vSpeed / 10.0;
                    if (pitch < -1)
                        pitch = -1;
                    if (pitch > 1)
                        pitch = 1;
                    pitch = 0.8 + (pitch + 1.0) * 0.3;
                    varioSound->setPitch(pitch);

                    float vol = (fabs((pitch - 1.0)) * 5.0) - 0.2;
                    if (vol > 1.0)
                        vol = 1.0;
                    if (vol < 0.0)
                        vol = 0.0;
                    varioSound->setVolume(vol);
                    if (vol > 1.0)
                        vol = 1.0;


                    Winds->SetWindNED(currentVelocity.y(), currentVelocity.x(), -currentVelocity.z());
                    Winds->SetTurbGain(currentTurbulence);
                    //fprintf(stderr, "cv: %f\n", currentVelocity.z());
                    targetVelocity.set(WX->number(), WY->number(), WZ->number());
                    targetTurbulence = 0;
                }
            }

        }

            coVRMSController::instance()->sendSlaves((char *)lastPos.ptr(), sizeof(lastPos));
                    VRSceneGraph::instance()->getTransform()->setMatrix(lastPos);
                    coVRCollaboration::instance()->SyncXform();
            return result;
}
else
{
            coVRMSController::instance()->readMaster((char *)lastPos.ptr(), sizeof(lastPos));
                    VRSceneGraph::instance()->getTransform()->setMatrix(lastPos);
                    coVRCollaboration::instance()->SyncXform();
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
        varioSound->play();
    }
    else
    {
        varioSound->stop();
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

bool
JSBSimPlugin::updateUdp()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
    if (udp)
    {
        std::cout << "udp = true" << endl;
        static bool firstTime = true;
	
        int status = 0;

	if(deviceVersion == 2)
	{
	    status = udp->receive(&gliderValues, sizeof(gliderValues), 0.0);
	//fprintf(stderr,"sizeof(gliderValues):%d status %d\n",(int)sizeof(gliderValues),status);
	}
	else
	{
	    status = udp->receive(&fgcontrol, sizeof(FGControl), 0.0);
	}

        if (status == sizeof(FGControl))
        {
            std::cout << "status == sizeof(FGControl)" << endl;
            byteSwap(fgcontrol.aileron);
            byteSwap(fgcontrol.elevator);
            std::cerr << "JSBSimPlugin::updateUdp:"<<  fgcontrol.aileron << "     " << fgcontrol.elevator<< std::endl;
        }
        if (status == sizeof(gliderValues))
        {
	
            /*byteSwap(gliderValues.left);
            byteSwap(gliderValues.right);
            byteSwap(gliderValues.angle);
            byteSwap(gliderValues.speed);
            byteSwap(gliderValues.state);*/

	    float leftLine = (gliderValues.left/1280.0); 
            float rightLine =( gliderValues.right/1280.0); 
	    float angleValue = -(gliderValues.angle/180.0); 
    if (leftLine>1.0)
       leftLine=1.0;
    if (leftLine<0.0)
       leftLine=0.0;
    if (rightLine>1.0)
       rightLine=1.0;
    if (rightLine<0.0)
       rightLine=0.0;
	    
            double elevatorMin=-1,elevatorMax=1,aileronMax=1;
	        fgcontrol.elevator=-((leftLine+rightLine)/2*(elevatorMax-elevatorMin)+elevatorMin);
            fgcontrol.aileron=(-leftLine+rightLine+angleValue)*aileronMax;


            std::cerr << "JSBSimPlugin::left:"<<  gliderValues.left << "     " << leftLine<<"     " << gliderValues.angle<< std::endl;
            std::cerr << "JSBSimPlugin::right:"<<  gliderValues.right << "     " << rightLine<<"     " << angleValue<< std::endl;
        }
        else if (status == -1)
        {
            if (firstTime)
            {
                std::cerr << "JSBSimPlugin::updateUdp: error while reading data" << std::endl;
                firstTime = false;
            }
            //initUDP();
            return false;
        }
        else
        {
            std::cerr << "JSBSimPlugin::updateUdp: received invalid no. of bytes: recv=" << status << ", got=" << status << std::endl;
            initUDP();
            return false;
        }
    }
    return true;
}

void JSBSimPlugin::initUDP()
{
    /*delete udp;

    std::cerr << "JSBSim config: UDP: serverHost: " << host << ", localPort: " << localPort << ", serverPort: " << serverPort << std::endl;
    udp = new UDPComm(host.c_str(), serverPort, localPort);*/
    return;
}

void JSBSimPlugin::addThermal(const osg::Vec3& velocity, float turbulence)
{
    targetVelocity += velocity;
    targetTurbulence += turbulence;
}

COVERPLUGIN(JSBSimPlugin)


void VrmlNodeThermal::initFields(VrmlNodeThermal* node, VrmlNodeType* t)
{
    initFieldsHelper(node, t,
    exposedField("direction", node->d_direction),
    exposedField("location", node->d_location),
    exposedField("maxBack", node->d_maxBack),
    exposedField("maxFront", node->d_maxFront),
    exposedField("minBack", node->d_minBack),
    exposedField("minFront", node->d_minFront),
    exposedField("height", node->d_height),
    exposedField("velocity", node->d_velocity),
    exposedField("turbulence", node->d_turbulence));
}

const char* VrmlNodeThermal::name()
{
    return "Thermal";
}

VrmlNodeThermal::VrmlNodeThermal(VrmlScene* scene)
    : VrmlNodeChild(scene, name())
    , d_direction(0, 0, 1)
    , d_location(0, 0, 0)
    , d_maxBack(10)
    , d_maxFront(10)
    , d_minBack(1)
    , d_minFront(1)
    , d_height(100)
    , d_velocity(0, 0, 4)
    , d_turbulence(0.0)
{
    numThermalNodes++;
}
int VrmlNodeThermal::numThermalNodes = 0;

VrmlNodeThermal::VrmlNodeThermal(const VrmlNodeThermal& n)
    : VrmlNodeChild(n)
{
    d_direction = n.d_direction;
    d_location = n.d_location;
    d_maxBack = n.d_maxBack;
    d_maxFront = n.d_maxFront;
    d_minBack = n.d_minBack;
    d_minFront = n.d_minFront;
    d_height = n.d_height;
    d_velocity = n.d_velocity;
    d_turbulence = n.d_turbulence;
    numThermalNodes++;
}

VrmlNodeThermal::~VrmlNodeThermal()
{
    numThermalNodes--;
}

void VrmlNodeThermal::eventIn(double timeStamp,
    const char* eventName,
    const VrmlField* fieldValue)
{
    //if (strcmp(eventName, "carNumber"))
    // {
    //}
    // Check exposedFields
    //else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

}

void VrmlNodeThermal::render(Viewer* viewer)
{
    // Is viewer inside the cylinder?
    float x, y, z;
    viewer->getPosition(&x, &y, &z);
    if (y > d_location.y() && y < d_location.y() + d_height.get())
    {
        VrmlSFVec3f toViewer(x, y, z);
        toViewer.subtract(&d_location); // now we have the vector to the viewer
        VrmlSFVec3f dir = d_direction;
        *(toViewer.get() + 1) = 0; // y == height = 0
        *(dir.get() + 1) = 0;
        float dist = (float)toViewer.length();
        toViewer.normalize();
        dir.normalize();
        // angle between the sound direction and the viewer
        float angle = (float)acos(toViewer.dot(&d_direction));
        //fprintf(stderr,"angle: %f",angle/M_PI*180.0);
        float cang = (float)cos(angle / 2.0);
        float rmin, rmax;
        double intensity;
        rmin = fabs(d_minBack.get() * d_minFront.get() / (cang * cang * (d_minBack.get() - d_minFront.get()) + d_minFront.get()));
        rmax = fabs(d_maxBack.get() * d_maxFront.get() / (cang * cang * (d_maxBack.get() - d_maxFront.get()) + d_maxFront.get()));
        //fprintf(stderr,"rmin: %f rmax: %f",rmin,rmax);
        if (dist <= rmin)
            intensity = 1.0;
        else if (dist > rmax)
            intensity = 0.0;
        else
        {
            intensity = (rmax - dist) / (rmax - rmin);
        }
        osg::Vec3 v(d_velocity.x(), -d_velocity.z(), d_velocity.y()); // velocities are in VRML orientation (y-up)
        v *= intensity;
        JSBSimPlugin::instance()->addThermal(v, d_turbulence.get() * intensity);
    }
    setModified();
}


