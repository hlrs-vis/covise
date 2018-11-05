/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "JSBSim.h"
#include "models/FGFCS.h"
#include "FGJSBBase.h"
#include "models/FGMassBalance.h"
#include "models/FGPropagate.h"
#include "cover/VRSceneGraph.h"
#include "cover/coVRCollaboration.h"
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
}

// this is called if the plugin is removed at runtime
JSBSimPlugin::~JSBSimPlugin()
{
    fprintf(stderr, "JSBSimPlugin::~JSBSimPlugin\n");

    delete FDMExec;

}

bool JSBSimPlugin::init()
{

    JSBMenu = new ui::Menu("JSBSim", this);

    printCatalog = new ui::Action(JSBMenu, "printCatalog");
    printCatalog->setCallback([this]() {
            FDMExec->PrintPropertyCatalog();
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
    FDMExec->Setdt(1.0 / 60.0);

    bool result = false;

    std::string line = coCoviseConfig::getEntry("COVER.Plugin.JSBSim.ScriptName");
    SGPath ScriptName;
    ScriptName.set(line);
    std::string AircraftDir = coCoviseConfig::getEntry("aircraftDir", "COVER.Plugin.JSBSim.Model","D:/src/gitbase/jsbsim/aircraft");
    std::string AircraftName = coCoviseConfig::getEntry("aircraft", "COVER.Plugin.JSBSim.Model","paraglider");
    std::string EnginesDir = coCoviseConfig::getEntry("enginesDir", "COVER.Plugin.JSBSim.Model", "D:/src/gitbase/jsbsim/aircraft/paraglider/Engines");
    std::string SystemsDir = coCoviseConfig::getEntry("aircraftDir", "COVER.Plugin.JSBSim.Model", "D:/src/gitbase/jsbsim/aircraft/paraglider/Systems");
    std::string resetFile = coCoviseConfig::getEntry("resetFile", "COVER.Plugin.JSBSim.Model", "D:/src/gitbase/jsbsim/aircraft/paraglider/reset00.xml");
    // *** OPTION A: LOAD A SCRIPT, WHICH LOADS EVERYTHING ELSE *** //
    if (!ScriptName.isNull()) {

        result = FDMExec->LoadScript(ScriptName, 1.0/60.0, ResetName);

        if (!result) {
            cerr << "Script file " << ScriptName << " was not successfully loaded" << endl;
            return false;
        }

        // *** OPTION B: LOAD AN AIRCRAFT AND A SET OF INITIAL CONDITIONS *** //
    }
    else if (!AircraftName.empty() || !ResetName.isNull()) {

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

    frame_duration = FDMExec->GetDeltaT();

    initial_seconds = cover->frameTime();


    FDMExec->RunIC();
    FDMExec->Setsim_time(cover->frameTime());

    result = FDMExec->Run();  // MAKE AN INITIAL RUN

    // PRINT SIMULATION CONFIGURATION
    FDMExec->PrintSimulationConfiguration();

    return true;
}

bool
JSBSimPlugin::update()
{

    FCS->SetDaCmd(0.0);
    FCS->SetDeCmd(0.0);

    bool result = false;
    
    current_seconds = cover->frameTime();
    double sim_lag_time = current_seconds - FDMExec->GetSimTime(); // How far behind sim-time is from actual
                                                                // elapsed time.
    //for (int i = 0; i<(int)(sim_lag_time / frame_duration); i++) {  // catch up sim time to actual elapsed time.
        result = FDMExec->Run();
    //    if (FDMExec->Holding()) break;
    //}


    if (cover->frameTime() > printTime + 5)
    {
        printTime = cover->frameTime();
        // Dump the simulation state (position, orientation, etc.)
        FDMExec->GetPropagate()->DumpState();
    }

    osg::Matrix rot;
    rot.makeRotate(Propagate->GetEuler(JSBSim::FGJSBBase::ePsi), osg::Vec3(0, 0, 1), Propagate->GetEuler(JSBSim::FGJSBBase::eTht), osg::Vec3(1, 0, 0),Propagate->GetEuler(JSBSim::FGJSBBase::ePhi), osg::Vec3(0, 1, 0) );
    osg::Matrix trans;
    trans.makeTranslate(-MassBalance->GetXYZcg(2),MassBalance->GetXYZcg(1), MassBalance->GetXYZcg(3));
    osg::Matrix Plane = osg::Matrix::inverse(rot* trans);


    VRSceneGraph::instance()->getTransform()->setMatrix(Plane);
    coVRCollaboration::instance()->SyncXform();
    return true;
}

COVERPLUGIN(JSBSimPlugin)
