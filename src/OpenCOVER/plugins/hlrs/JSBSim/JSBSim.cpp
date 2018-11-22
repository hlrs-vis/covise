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
#include "models/propulsion/FGEngine.h"
#include "models/propulsion/FGPiston.h"
#include "cover/VRSceneGraph.h"
#include "cover/coVRCollaboration.h"
#include <UDPComm.h>
#include <util/byteswap.h>
#include <util/unixcompat.h>

JSBSimPlugin *JSBSimPlugin::plugin = NULL;

JSBSimPlugin::JSBSimPlugin(): ui::Owner("JSBSimPlugin", cover->ui)
{
    fprintf(stderr, "JSBSimPlugin::JSBSimPlugin\n");
#if defined(_MSC_VER)
    // _clearfp();
    // _controlfp(_controlfp(0, 0) & ~(_EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW),
    //     _MCW_EM);
#elif defined(__GNUC__) && !defined(sgi) && !defined(__APPLE__)
    feenableexcept(FE_DIVBYZERO | FE_INVALID);
#endif

    plugin = this;
    udp = nullptr;
}

// this is called if the plugin is removed at runtime
JSBSimPlugin::~JSBSimPlugin()
{
    fprintf(stderr, "JSBSimPlugin::~JSBSimPlugin\n");

    delete FDMExec;

}

void JSBSimPlugin::reset(double dz)
{
    FDMExec->Setdt(1.0 / 120.0);
    frame_duration = FDMExec->GetDeltaT();

    FDMExec->Setsim_time(0.0);
    SimStartTime = cover->frameTime();
    osg::Vec3 viewerPosInFeet = cover->getInvBaseMat().getTrans() / 0.3048;

    double radius = (il.GetRadius() + viewerPosInFeet[2]);
    double ecX, ecY, ecZ;
    ecX = viewerPosInFeet[2] + radius;
    ecY = -viewerPosInFeet[1];
    ecZ = viewerPosInFeet[0];
    double r02 = ecX * ecX + ecY * ecY;
    double rxy = sqrt(r02);
    double mLon, mLat;
    eyePoint.makeTranslate(Aircraft->GetXYZep(1)*25.4, Aircraft->GetXYZep(2)*25.4, Aircraft->GetXYZep(3)*25.4);


    // Compute the longitude and latitude itself
    if (ecX == 0.0 && ecY == 0.0)
        mLon = 0.0;
    else
        mLon = atan2(ecY, ecX);

    if (rxy == 0.0 && ecZ == 0.0)
        mLat = 0.0;
    else
        mLat = atan2(ecZ, rxy);
    osg::Matrix viewer = cover->getInvBaseMat();
    osg::Vec3 dir;
    for (int i = 0; i < 3; i++)
        dir[i] = viewer(1, i);
    osg::Vec3 y(0, 1, 0);
    dir.normalize();


    FDMExec->GetIC()->SetLatitudeRadIC(mLat);
    FDMExec->GetIC()->SetLongitudeRadIC(mLon);
    FDMExec->GetIC()->SetAltitudeASLFtIC(viewerPosInFeet[2] - Aircraft->GetXYZep(3)+dz/0.3048);
    FDMExec->GetIC()->SetPsiRadIC(atan2(y[1], y[0]) - atan2(dir[1], dir[0]) - M_PI_2);
    FDMExec->GetIC()->SetThetaRadIC(0.0);
    FDMExec->GetIC()->SetPhiRadIC(0.0);



    Propagate->InitModel();
    FDMExec->RunIC();
}

bool JSBSimPlugin::init()
{
    delete udp;

    const std::string host = covise::coCoviseConfig::getEntry("value", "JSBSim.serverHost", "141.58.8.212");
    unsigned short serverPort = covise::coCoviseConfig::getInt("JSBSim.serverPort", 1234);
    unsigned short localPort = covise::coCoviseConfig::getInt("JSBSim.localPort", 5252);
    std::cerr << "JSBSim config: UDP: serverHost: " << host << ", localPort: " << localPort << ", serverPort: " << serverPort << std::endl;
    udp = new UDPComm(host.c_str(), serverPort, localPort);

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

    resetButton = new ui::Action(JSBMenu, "reset");
    resetButton->setCallback([this]() {
        if (FDMExec)
            reset();
    });

    upButton = new ui::Action(JSBMenu, "Up");
    upButton->setCallback([this]() {
        if (FDMExec)
            reset(100);
    });

    // *** SET UP JSBSIM *** //
    FDMExec = new JSBSim::FGFDMExec();
    FDMExec->SetRootDir(RootDir);
    FDMExec->SetAircraftPath(SGPath("aircraft"));
    FDMExec->SetEnginePath(SGPath("engine"));
    FDMExec->SetSystemsPath(SGPath("systems"));
    FDMExec->GetPropertyManager()->Tie("simulation/frame_start_time", &actual_elapsed_time);
    FDMExec->GetPropertyManager()->Tie("simulation/cycle_duration", &cycle_duration);


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

    if (nohighlight) FDMExec->disableHighLighting();
    
    bool result = false;

    std::string line = coCoviseConfig::getEntry("COVER.Plugin.JSBSim.ScriptName");
    SGPath ScriptName;
    ScriptName.set(line);
    std::string AircraftDir = coCoviseConfig::getEntry("aircraftDir", "COVER.Plugin.JSBSim.Model", "D:/src/gitbase/jsbsim/aircraft");
    std::string AircraftName = coCoviseConfig::getEntry("aircraft", "COVER.Plugin.JSBSim.Model", "paraglider");
    std::string EnginesDir = coCoviseConfig::getEntry("enginesDir", "COVER.Plugin.JSBSim.Model", "D:/src/gitbase/jsbsim/aircraft/paraglider/Engines");
    std::string SystemsDir = coCoviseConfig::getEntry("systemsDir", "COVER.Plugin.JSBSim.Model", "D:/src/gitbase/jsbsim/aircraft/paraglider/Systems");
    std::string resetFile = coCoviseConfig::getEntry("resetFile", "COVER.Plugin.JSBSim.Model", "D:/src/gitbase/jsbsim/aircraft/paraglider/reset00.xml");
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

        JSBSim::FGInitialCondition *IC = FDMExec->GetIC();
        if (!IC->Load(SGPath(resetFile))) {
            cerr << "Initialization unsuccessful" << endl;
            return false;
        }
    }
    else {
        cout << "  No Aircraft, Script, or Reset information given" << endl << endl;
        return false;
    }

    il.SetLatitude(0.0);
    il.SetLongitude(0.0);
    il.SetAltitudeASL(0.0);

    fgcontrol.aileron = 0.0;
    fgcontrol.elevator = 0.0;

    reset(0.0);

    
    // PRINT SIMULATION CONFIGURATION
    //FDMExec->PrintSimulationConfiguration();
    

    return true;
}

void JSBSimPlugin::key(int type, int keySym, int mod)
{
    (void)mod;
    switch (type)
    {
    case (osgGA::GUIEventAdapter::KEYDOWN):
        if (keySym == 'l' || keySym == 65363)
        {
            if(fgcontrol.aileron < 0.99)
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
        break;
    }
    fprintf(stderr, "Keysym: %d\n", keySym);
    fprintf(stderr, "Aileron: %lf Elevator %lf\n", fgcontrol.aileron,fgcontrol.elevator);
}

bool
JSBSimPlugin::update()
{
    updateUdp();

    FCS->SetDaCmd(fgcontrol.aileron);
    FCS->SetDeCmd(fgcontrol.elevator);

    for (unsigned int i = 0; i < Propulsion->GetNumEngines(); i++) {
        FCS->SetThrottleCmd(i,1.0);
        FCS->SetMixtureCmd(i,1.0);

        switch (Propulsion->GetEngine(i)->GetType())
        {
        case JSBSim::FGEngine::etPiston:
        { // FGPiston code block
            JSBSim::FGPiston* eng = (JSBSim::FGPiston*)Propulsion->GetEngine(i);
            eng->SetMagnetos(3);
            break;
        } // end FGPiston code block
        }
        { // FGEngine code block
            JSBSim::FGEngine* eng = Propulsion->GetEngine(i);
            eng->SetStarter(1);
            eng->SetRunning(1);
        } // end FGEngine code block
    }
    bool result = false;

    while (FDMExec->GetSimTime() + SimStartTime < cover->frameTime())
    {
        if (FDMExec->Holding()) break;
        result = FDMExec->Run();
    }

    //if (cover->frameTime() > printTime + 1)
    //{
    //    printTime = cover->frameTime();
        // Dump the simulation state (position, orientation, etc.)
        //FDMExec->GetPropagate()->DumpState();
    //}

    osg::Matrix rot;
    rot.makeRotate(Propagate->GetEuler(JSBSim::FGJSBBase::ePsi), osg::Vec3(0, 0, -1), Propagate->GetEuler(JSBSim::FGJSBBase::eTht) , osg::Vec3(0, 1, 0), Propagate->GetEuler(JSBSim::FGJSBBase::ePhi), osg::Vec3(1, 0, 0));
    osg::Matrix trans;

    JSBSim::FGPropagate::VehicleState location = Propagate->GetVState();
    float scale = cover->getScale();
    trans.makeTranslate(location.vLocation(3)*0.3048*scale, -location.vLocation(2)*0.3048*scale, (location.vLocation(1) - il.GetRadius())*0.3048*scale);
    osg::Matrix preRot;
    preRot.makeRotate(-M_PI_2, 0, 0, 1.0);
    osg::Matrix Plane = osg::Matrix::inverse(preRot*eyePoint*rot*trans);
    /*JSBSim::FGMatrix33 tb2lMat = Propagate->GetTb2l();
    Plane.makeIdentity();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            Plane(i, j) = tb2lMat.Entry(i+1, j+1);
     Plane.invert(Plane); */

    if (FDMExec && !FDMExec->Holding())
    {
        VRSceneGraph::instance()->getTransform()->setMatrix(Plane);
        coVRCollaboration::instance()->SyncXform();
    }
    return true;
}

bool
JSBSimPlugin::updateUdp()
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
    if (udp)
    {
        int status = udp->receive(&fgcontrol, sizeof(FGControl),0.001);

        if (status == sizeof(FGControl))
        {
            byteSwap(fgcontrol.aileron);
            byteSwap(fgcontrol.elevator);
            FCS->SetDaCmd(fgcontrol.aileron);
            FCS->SetDeCmd(fgcontrol.elevator);
        }
        else if (status == -1)
        {
            std::cerr << "FlightGear::update: error while reading data" << std::endl;
            initUDP();
            return false;
        }
        else
        {
            std::cerr << "FlightGear::update: received invalid no. of bytes: recv=" << status << ", got=" << status << std::endl;
            initUDP();
            return false;
        }
    }
    return true;
}

void JSBSimPlugin::initUDP()
{
    delete udp;

    const std::string host = covise::coCoviseConfig::getEntry("value", "JSBSim.serverHost", "141.58.8.212");
    unsigned short serverPort = covise::coCoviseConfig::getInt("JSBSim.serverPort", 1234);
    unsigned short localPort = covise::coCoviseConfig::getInt("JSBSim.localPort", 5252);
    std::cerr << "JSBSim config: UDP: serverHost: " << host << ", localPort: " << localPort << ", serverPort: " << serverPort << std::endl;
    udp = new UDPComm(host.c_str(), serverPort, localPort);
    return;
}

COVERPLUGIN(JSBSimPlugin)
