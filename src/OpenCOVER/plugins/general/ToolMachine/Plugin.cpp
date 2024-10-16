#include "Plugin.h"
#include "VrmlNode.h"

#include <util/coExport.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Menu.h>
#include <cover/VRViewer.h>

#include <stdlib.h>
#include <cover/ui/VectorEditField.h>

#include <OpcUaClient/opcua.h>
#include <osgDB/ReadFile>


using namespace covise;
using namespace opencover;
using namespace vrml;


COVERPLUGIN(ToolMaschinePlugin)

ToolMaschinePlugin::ToolMaschinePlugin()
:coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ToolMachinePlugin", cover->ui)
, m_menu(new ui::Menu("ToolMachine", this))
, m_pauseBtn(new ui::Button(m_menu, "pause"))
{
    m_menu->allowRelayout(true);
    VrmlNamespace::addBuiltIn(MachineNode::defineType());
    VrmlNamespace::addBuiltIn(MachineNodeArrayMode::defineType());
    VrmlNamespace::addBuiltIn(MachineNodeSingleMode::defineType());
    VrmlNamespace::addBuiltIn(ToolChangerNode::defineType());
    std::cerr << "added vrml nodes" << "MachineNode, MachineNodeArrayMode, MachineNodeSingleMode, ToolChangerNode" << std::endl;
    config()->setSaveOnExit(true);
    

    // m_offsets = new opencover::ui::VectorEditField(menu, "offsetInMM");
    // m_offsets->setValue(osg::Vec3(-406.401596,324.97962,280.54943));
    std::array<std::string, 6> names = {"x", "y", "z", "a", "b", "c"};
    for (size_t i = 0; i < 6; i++)
    {
        auto slider = new ui::Slider(m_menu, names[i] + "Pos");
        i < 3 ? slider->setBounds(-200, 200) : slider->setBounds(0, 360);
        
        slider->setCallback([i, this](double val, bool b){
            m_pauseMove = !b;
            if(m_machine)
                m_machine->move(i, val);
        });
    }
    m_pauseBtn->setCallback([this](bool state){
        if(m_machine)
            m_machine->pause(state);
    });
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
    if(!m_machine && machineNodes.size() > 0)
    {
        m_machine = std::make_unique<Machine>(machineNodes[0]);
        m_machine->setUi(m_menu, config().get());
    }

    if(!m_toolChanger && toolChangers.size() > 0 && toolChangers[0]->allInitialized())
    {
        m_toolChanger = std::make_unique<ToolChanger>(m_menu, config().get(), ToolChangerFiles{toolChangers[0]->arm->get(), toolChangers[0]->changer->get(), toolChangers[0]->cover->get()}, nullptr);
    }

    if(m_pauseMove || m_pauseBtn->state())
        return true;
    
    if(!m_toolHeadSet && m_machine && m_toolChanger)
    {
        m_toolChanger->setToolHead(m_machine->getToolHead());
        m_toolHeadSet = true;
    }
    
    if(m_machine)
        m_machine->update();
    if(m_toolChanger)
        m_toolChanger->update();

    return true;
}

