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

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>
#include <config/CoviseConfig.h>
#include <util/byteswap.h>
#include <net/covise_connect.h>

#include "initialization/FGTrim.h"
#include "FGFDMExec.h"
#include "input_output/FGXMLFileRead.h"

#include <util/coTypes.h>
#include <cover/ui/Button.h>
#include <cover/ui/Action.h>
#include <cover/ui/Menu.h>

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

    SGPath RootDir;
    SGPath ScriptName;
    string AircraftName;
    SGPath ResetName;
    vector <string> LogOutputName;
    vector <SGPath> LogDirectiveName;
    vector <string> CommandLineProperties;
    vector <double> CommandLinePropertyValues;
    JSBSim::FGFDMExec* FDMExec;
    JSBSim::FGTrim* trimmer;

    bool realtime;
    bool play_nice;
    bool suspend;
    bool catalog;
    bool nohighlight;

    double end_time = 1e99;
    double actual_elapsed_time = 0;
    double cycle_duration = 0;
};
#endif
