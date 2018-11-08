/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _JSBSim_NODE_PLUGIN_H
#define _JSBSim_NODE_PLUGIN_H

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
class UDPComm;

using JSBSim::FGXMLFileRead;
using JSBSim::Element;
using namespace opencover;
using namespace covise;

class JSBSimPlugin : public coVRPlugin, public ui::Owner
{
public:
    JSBSimPlugin();
    ~JSBSimPlugin();
    bool init();
    static JSBSimPlugin *plugin;

    bool update();

private:
    ui::Menu *JSBMenu;
    ui::Action *printCatalog;
    ui::Button *pauseButton;

    SGPath RootDir;
    SGPath ScriptName;
    string AircraftName;
    SGPath ResetName;
    vector <string> LogOutputName;
    vector <SGPath> LogDirectiveName;
    vector <string> CommandLineProperties;
    vector <double> CommandLinePropertyValues;

    double current_seconds = 0.0;
    double initial_seconds = 0.0;
    double frame_duration = 0.0;
    double printTime = 0.0;

    JSBSim::FGFDMExec* FDMExec;
    JSBSim::FGTrim* trimmer;

    JSBSim::FGAtmosphere*      Atmosphere;
    JSBSim::FGWinds*           Winds;
    JSBSim::FGFCS*             FCS;
    JSBSim::FGPropulsion*      Propulsion;
    JSBSim::FGMassBalance*     MassBalance;
    JSBSim::FGAircraft*        Aircraft;
    JSBSim::FGPropagate*       Propagate;
    JSBSim::FGAuxiliary*       Auxiliary;
    JSBSim::FGAerodynamics*    Aerodynamics;
    JSBSim::FGGroundReactions* GroundReactions;
    JSBSim::FGInertial*        Inertial;
    JSBSim::FGAccelerations*   Accelerations;

    JSBSim::FGPropagate::VehicleState initialLocation;

    osg::Vec3d zeroPosition;

    struct FGControl
    {
        double elevator;
        double aileron;
    } fgcontrol;
    UDPComm *udp;
    void initUDP();
    bool updateUdp();

    bool realtime;
    bool play_nice;
    bool suspend;
    bool catalog;
    bool nohighlight;

    double end_time = 1e99;
    double actual_elapsed_time = 0;
    double cycle_duration = 0;
    OpenThreads::Mutex mutex;
};
#endif
