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

JSBSimPlugin* JSBSimPlugin::plugin = NULL;

JSBSimPlugin::JSBSimPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("JSBSimPlugin", cover->ui)
, coVRNavigationProvider("Paraglider", this)
{
    fprintf(stderr, "JSBSimPlugin::JSBSimPlugin\n");
    
    const char* GF = coVRFileManager::instance()->getName("share/covise/jsbsim/geometry/paraglider.osgb");
    if (GF == nullptr)
        GF = "";
    geometryFile = coCoviseConfig::getEntry("geometry", "COVER.Plugin.JSBSim.Geometry", GF);
    float tx = coCoviseConfig::getFloat("x", "COVER.Plugin.JSBSim.Geometry", 0.0);
    float ty = coCoviseConfig::getFloat("y", "COVER.Plugin.JSBSim.Geometry", 0.0);
    float tz = coCoviseConfig::getFloat("z", "COVER.Plugin.JSBSim.Geometry", 0.0);
    float th = coCoviseConfig::getFloat("h", "COVER.Plugin.JSBSim.Geometry", 0.0);
    float tp = coCoviseConfig::getFloat("p", "COVER.Plugin.JSBSim.Geometry", 0.0);
    float tr = coCoviseConfig::getFloat("r", "COVER.Plugin.JSBSim.Geometry", 0.0);
    float ts = coCoviseConfig::getFloat("scale", "COVER.Plugin.JSBSim.Geometry", 1000.0);
    osg::Matrix gt = osg::Matrix::scale(ts,ts,ts)*osg::Matrix::rotate(th, osg::Vec3(0,0,1), tp, osg::Vec3(1,0,0),  tr, osg::Vec3(0,1,0)) *  osg::Matrix::translate(tx, ty, tz);
    geometryTrans = new osg::MatrixTransform(gt);
    osg::Node* n = osgDB::readNodeFile(geometryFile.c_str());
    if(n!=nullptr)
    {
        n->setName(geometryFile);
        geometryTrans->addChild(n);
    }
    else
    {
        cerr << "could not load Geometry file " << geometryFile << endl;
    }
if (coVRMSController::instance()->isMaster())
        {
    remoteSoundServer = coCoviseConfig::getEntry("server", "COVER.Plugin.JSBSim.Sound", "localhost");
    remoteSoundPort = coCoviseConfig::getInt("port", "COVER.Plugin.JSBSim.Sound", 31805);
    const char* VS = coVRFileManager::instance()->getName("share/covise/jsbsim/Sounds/vario.wav");
    if (VS == nullptr)
        VS = "";
    VarioSound = coCoviseConfig::getEntry("vario", "COVER.Plugin.JSBSim.Sound", VS);
    const char* WS = coVRFileManager::instance()->getName("share/covise/jsbsim/Sounds/wind1.wav");
    if (WS == nullptr)
        WS = "";
    WindSound = coCoviseConfig::getEntry("wind", "COVER.Plugin.JSBSim.Sound", WS);
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
    udp = nullptr;
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

    std::string line = coCoviseConfig::getEntry("COVER.Plugin.JSBSim.ScriptName");
    ScriptName.set(line);
    AircraftDir = coCoviseConfig::getEntry("aircraftDir", "COVER.Plugin.JSBSim.Model", "aircraft");
    AircraftName = coCoviseConfig::getEntry("aircraft", "COVER.Plugin.JSBSim.Model", "paraglider");
    EnginesDir = coCoviseConfig::getEntry("enginesDir", "COVER.Plugin.JSBSim.Model", "engines");
    SystemsDir = coCoviseConfig::getEntry("systemsDir", "COVER.Plugin.JSBSim.Model", "paraglider/Systems");
    resetFile = coCoviseConfig::getEntry("resetFile", "COVER.Plugin.JSBSim.Model", "reset00.xml");
    // *** OPTION A: LOAD A SCRIPT, WHICH LOADS EVERYTHING ELSE *** //
    if (!ScriptName.isNull()) {

        result = FDMExec->LoadScript(ScriptName, 0.008333, SGPath(resetFile));

        if (!result) {
            cerr << "Script file " << ScriptName << " was not successfully loaded" << endl;
            return false;
        }

        // *** OPTION B: LOAD AN AIRCRAFT AND A SET OF INITIAL CONDITIONS *** //
    }
    else if (!AircraftName.empty() || !resetFile.length() == 0) {

        if (catalog) FDMExec->SetDebugLevel(0);

        if (!FDMExec->LoadModel(SGPath(AircraftDir),
            SGPath(EnginesDir),
            SGPath(SystemsDir),
            AircraftName)) {
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

    host = covise::coCoviseConfig::getEntry("host", "COVER.Plugin.JSBSim.Glider", "141.58.8.212");
    serverPort = covise::coCoviseConfig::getInt("serverPort","COVER.Plugin.JSBSim.Glider", 31319);
    localPort = covise::coCoviseConfig::getInt("localPort","COVER.Plugin.JSBSim.Glider", 1234);
    const char* rd = coVRFileManager::instance()->getName("share/covise/jsbsim");
    if(rd==nullptr)
        rd="";
    RootDir = covise::coCoviseConfig::getEntry("rootDir", "COVER.Plugin.JSBSim", rd).c_str();

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

    std::string proj_from = coCoviseConfig::getEntry("from", "COVER.Plugin.JSBSim.Projection", "+proj=latlong +datum=WGS84");
    if (!(pj_from = pj_init_plus(proj_from.c_str())))
    {
        fprintf(stderr, "ERROR: pj_init_plus failed with pj_from = %s\n", proj_from.c_str());
    }

    std::string proj_to = coCoviseConfig::getEntry("to", "COVER.Plugin.JSBSim.Projection", "+proj=tmerc +lat_0=0 +lon_0=9 +k=1.000000 +x_0=9703.397 +y_0=-5384244.453 +ellps=bessel +datum=potsdam");// +nadgrids=" + dir + std::string("BETA2007.gsb");

    if (!(pj_to = pj_init_plus(proj_to.c_str())))
    {
        fprintf(stderr, "ERROR: pj_init_plus failed with pj_to = %s\n", proj_to.c_str());
    }
    projectOffset[0] = coCoviseConfig::getFloat("offetX", "COVER.Plugin.JSBSim.Projection", 0);
    projectOffset[1] = coCoviseConfig::getFloat("offetY", "COVER.Plugin.JSBSim.Projection", 0);
    projectOffset[2] = coCoviseConfig::getFloat("offetZ", "COVER.Plugin.JSBSim.Projection", 0);

    JSBMenu = new ui::Menu("JSBSim", this);

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

    VrmlNamespace::addBuiltIn(VrmlNodeThermal::defineType());
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
if (coVRMSController::instance()->isMaster())
        {
    updateUdp();
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
            FCS->SetDeCmd(fgcontrol.elevator);

            for (unsigned int i = 0; i < Propulsion->GetNumEngines(); i++) {
                FCS->SetThrottleCmd(i, 1.0);
                FCS->SetMixtureCmd(i, 1.0);

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
                rot.makeRotate(Propagate->GetEuler(JSBSim::FGJSBBase::ePsi), osg::Vec3(0, 0, -1), Propagate->GetEuler(JSBSim::FGJSBBase::eTht), osg::Vec3(0, 1, 0), Propagate->GetEuler(JSBSim::FGJSBBase::ePhi), osg::Vec3(1, 0, 0));
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
        static bool firstTime = true;
        int status = udp->receive(&fgcontrol, sizeof(FGControl), 0.0);

        if (status == sizeof(FGControl))
        {
            byteSwap(fgcontrol.aileron);
            byteSwap(fgcontrol.elevator);
            std::cerr << "JSBSimPlugin::updateUdp:"<<  fgcontrol.aileron << "     " << fgcontrol.elevator<< std::endl;
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


static VrmlNode* creator(VrmlScene* scene)
{
    return new VrmlNodeThermal(scene);
}

// Define the built in VrmlNodeType:: "Thermal" fields

VrmlNodeType* VrmlNodeThermal::defineType(VrmlNodeType* t)
{
    static VrmlNodeType* st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Thermal", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class



    t->addExposedField("direction", VrmlField::SFVEC3F);
    t->addExposedField("intensity", VrmlField::SFFLOAT);
    t->addExposedField("location", VrmlField::SFVEC3F);
    t->addExposedField("maxBack", VrmlField::SFFLOAT);
    t->addExposedField("maxFront", VrmlField::SFFLOAT);
    t->addExposedField("minBack", VrmlField::SFFLOAT);
    t->addExposedField("minFront", VrmlField::SFFLOAT);
    t->addExposedField("height", VrmlField::SFFLOAT);
    t->addExposedField("velocity", VrmlField::SFVEC3F);
    t->addExposedField("turbulence", VrmlField::SFFLOAT);

    return t;
}

VrmlNodeType* VrmlNodeThermal::nodeType() const
{
    return defineType(0);
}

VrmlNodeThermal::VrmlNodeThermal(VrmlScene* scene)
    : VrmlNodeChild(scene)
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
    : VrmlNodeChild(n.d_scene)
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

VrmlNode* VrmlNodeThermal::cloneMe() const
{
    return new VrmlNodeThermal(*this);
}

ostream& VrmlNodeThermal::printFields(ostream& os, int indent)
{
    return os;
}

// Set the value of one of the node fields.

void VrmlNodeThermal::setField(const char* fieldName,
    const VrmlField& fieldValue)
{

    if
        TRY_FIELD(direction, SFVec3f)
    else if
        TRY_FIELD(location, SFVec3f)
    else if
        TRY_FIELD(maxBack, SFFloat)
    else if
        TRY_FIELD(maxFront, SFFloat)
    else if
        TRY_FIELD(minBack, SFFloat)
    else if
        TRY_FIELD(minFront, SFFloat)
    else if
        TRY_FIELD(height, SFFloat)
    else if
        TRY_FIELD(velocity, SFVec3f)
    else if
        TRY_FIELD(turbulence, SFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
}

const VrmlField* VrmlNodeThermal::getField(const char* fieldName)
{
    if (strcmp(fieldName, "direction") == 0)
        return &d_direction;
    else if (strcmp(fieldName, "location") == 0)
        return &d_location;
    else if (strcmp(fieldName, "maxBack") == 0)
        return &d_maxBack;
    else if (strcmp(fieldName, "maxFront") == 0)
        return &d_maxFront;
    else if (strcmp(fieldName, "minBack") == 0)
        return &d_minBack;
    else if (strcmp(fieldName, "minFront") == 0)
        return &d_minFront;
    else if (strcmp(fieldName, "height") == 0)
        return &d_height;
    else if (strcmp(fieldName, "velocity") == 0)
        return &d_velocity;
    else if (strcmp(fieldName, "turbulence") == 0)
        return &d_turbulence;
    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
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


