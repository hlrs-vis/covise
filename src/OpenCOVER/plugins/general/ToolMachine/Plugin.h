/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVER_PLUGIN_TOOL_MASCHIE_H
#define COVER_PLUGIN_TOOL_MASCHIE_H

#include "Currents.h"
#include "Oct.h"
#include "ToolMachine.h"
#include "ToolChanger/ToolChanger.h"

#include <cover/coVRPluginSupport.h>
#include <cover/ui/Button.h>
#include <cover/ui/Owner.h>
#include <memory>
#include <open62541/client.h>
#include <osg/Vec3>

class ToolMaschinePlugin : public opencover::coVRPlugin, opencover::ui::Owner
{
public:
    ToolMaschinePlugin();
private:
    bool update() override;
    opencover::ui::Menu *m_menu;
    opencover::ui::Button *m_pauseBtn;
    bool m_pauseMove = false;
    std::unique_ptr<Machine> m_machine;
    std::unique_ptr<ToolChanger> m_toolChanger;
    bool m_toolHeadSet = false;
};




#endif
