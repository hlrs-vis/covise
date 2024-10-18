#include "Plugin.h"
#include "VrmlNode.h"
#include "ToolMachine.h"
#include "ToolChanger/ToolChanger.h"

#include <util/coExport.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Menu.h>
#include <cover/VRViewer.h>

#include <stdlib.h>
#include <cover/ui/VectorEditField.h>

#include <OpcUaClient/opcua.h>
#include <osgDB/ReadFile>

#include <vrml97/vrml/VrmlNamespace.h>
#include "ToolChanger/Tool.h"

using namespace covise;
using namespace opencover;
using namespace vrml;


COVERPLUGIN(ToolMaschinePlugin)

ToolMaschinePlugin::ToolMaschinePlugin()
:coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ToolMachinePlugin", cover->ui)
, m_menu(new ui::Menu("ToolMachine", this))
{
    m_menu->allowRelayout(true);
    VrmlNamespace::addBuiltIn(MachineNode::defineType());
    VrmlNamespace::addBuiltIn(MachineNodeArrayMode::defineType());
    VrmlNamespace::addBuiltIn(MachineNodeSingleMode::defineType());
    VrmlNamespace::addBuiltIn(ToolChangerNode::defineType());
    std::cerr << "added vrml nodes" << "MachineNode, MachineNodeArrayMode, MachineNodeSingleMode, ToolChangerNode" << std::endl;
    config()->setSaveOnExit(true);

}

osg::Vec3 toOsg(VrmlSFVec3f &v)
{
    return osg::Vec3(v.x(), v.y(), v.z());
}

osg::Quat toOsg(VrmlSFRotation &r)
{
    return osg::Quat(r.r(), osg::Vec3{r.x(), r.y(), r.z()});
}

bool ToolMaschinePlugin::update()
{
    for(auto machine : machineNodes)
    {
        if(!machine->machine)
            machine->machine = std::make_unique<Machine>(m_menu, config().get(), machine);
    }
    for(auto toolChanger : toolChangers)
    {
        if(!toolChanger->toolChanger)
            toolChanger->toolChanger = std::make_unique<ToolChanger>(m_menu, config().get(), toolChanger);
    }
    
    
    for(auto &m : machineNodes)
        m->machine->update();
    for(auto &t : toolChangers)
        t->toolChanger->update();

    return true;
}
