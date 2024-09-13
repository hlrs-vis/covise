#include "Plugin.h"


#include <util/coExport.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Menu.h>
#include <cover/VRViewer.h>

#include <stdlib.h>
#include <cover/ui/VectorEditField.h>

#include <OpcUaClient/opcua.h>


using namespace covise;
using namespace opencover;
using namespace vrml;


COVERPLUGIN(ToolMaschinePlugin)

ToolMaschinePlugin::ToolMaschinePlugin()
:coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ToolMachinePlugin", cover->ui)
, m_menu(new ui::Menu("ToolMachine", this))
, m_pauseBtn(new ui::Button(m_menu, "pause"))
, m_updateMode(std::make_unique<opencover::ui::SelectionListConfigValue>(m_menu, "updateMode", 0, *config(), "ToolMachinePlugin"))
{
    m_menu->allowRelayout(true);
    return;
    VrmlNamespace::addBuiltIn(MachineNode::defineType());

    config()->setSaveOnExit(true);
    

    // m_offsets = new opencover::ui::VectorEditField(menu, "offsetInMM");
    // m_offsets->setValue(osg::Vec3(-406.401596,324.97962,280.54943));
    std::array<std::string, 6> names = {"x", "y", "z", "a", "b", "c"};
    for (size_t i = 0; i < 6; i++)
    {
        auto slider = new ui::Slider(m_menu, names[i] + "Pos");
        i < 3 ? slider->setBounds(-100, 100) : slider->setBounds(0, 360);
        
        slider->setCallback([i, this](double val, bool b){
            m_pauseMove = !b;
            for (auto m : machineNodes)
                m->move(i, val);
        });
    }
    m_pauseBtn->setCallback([this](bool state){
        for(auto &m : machineNodes)
            m->pause(state);
    });

    const std::vector<std::string> updateMode{"all", "allOncePerFrame", "updatedOncePerFrame"};
    m_updateMode->ui()->setList(updateMode);
    m_updateMode->ui()->select(m_updateMode->getValue());
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
    if(m_pauseMove || m_pauseBtn->state())
        return true;
    for (const auto m : machineNodes)
    {
        m->setUi(m_menu, config().get());
        m->update(static_cast<UpdateMode>(m_updateMode->getValue()));
    }

    return true;
}

