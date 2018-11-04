/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "JSBSim.h"
JSBSimPlugin *JSBSimPlugin::plugin = NULL;

JSBSimPlugin::JSBSimPlugin(): ui::Owner("JSBSimPlugin", cover->ui)
{
    fprintf(stderr, "JSBSimPlugin::JSBSimPlugin\n");
#if defined(_MSC_VER)
    _clearfp();
    _controlfp(_controlfp(0, 0) & ~(_EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW),
        _MCW_EM);
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
    delete printCatalog;
    delete JSBMenu;

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

    if (nohighlight) FDMExec->disableHighLighting();
    FDMExec->Setdt(1.0 / 60.0);

    bool result = false;

    std::string line = coCoviseConfig::getEntry("COVER.Plugin.JSBSim.ScriptName");
    SGPath ScriptName;
    ScriptName.set(line);
    std::string AircraftName = coCoviseConfig::getEntry("COVER.Plugin.JSBSim.AircraftName");
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

        if (!FDMExec->LoadModel(SGPath("aircraft"),
            SGPath("engine"),
            SGPath("systems"),
            AircraftName)) {
            cerr << "  JSBSim could not be started" << endl << endl;
            return false;
        }


        JSBSim::FGInitialCondition *IC = FDMExec->GetIC();
        if (!IC->Load(ResetName)) {
            cerr << "Initialization unsuccessful" << endl;
            return false;
        }

    }
    else {
        cout << "  No Aircraft, Script, or Reset information given" << endl << endl;
        return false;
    }
    return true;
}

bool
JSBSimPlugin::update()
{
    
	return true;
}

COVERPLUGIN(JSBSimPlugin)
