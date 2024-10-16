#include "ToolMachine.h"
#include "Currents.h"
#include "Oct.h"

#include <vrml97/vrml/VrmlScene.h>

using namespace covise;
using namespace opencover;
using namespace vrml;

osg::MatrixTransform *toOsg(VrmlNode *node)
{
    auto g = node->toGroup();
    if(!g)
        return nullptr;
    auto vo = g->getViewerObject();
    if(!vo)
        return nullptr;
    auto pNode = ((osgViewerObject *)vo)->pNode;
    if(!pNode)
        return nullptr;
    auto trans = pNode->asTransform();
    if(!trans)
        return nullptr;
    return trans->asMatrixTransform();
}

Machine::Machine(MachineNodeBase *node)
: m_machineNode(node)
{}


void Machine::connectOpcua()
{

    if(!m_client)
        m_client = opcua::connect(m_machineNode->MachineName->get());
    if(!m_client || !m_client->isConnected())
        return;
    m_mathExpressionObserver = std::make_unique<MathExpressionObserver>(m_client);
    m_rdy = true;
    auto arrayMode = dynamic_cast<MachineNodeArrayMode *>(m_machineNode);
    auto singleMode = dynamic_cast<MachineNodeSingleMode *>(m_machineNode);

    if(arrayMode)
    {
        m_valueIds.push_back(m_client->observeNode(arrayMode->OPCUAArrayName->get()));
    }  else if (singleMode)
    {
        m_axisValueHandles.clear();
        for (size_t i = 0; i < singleMode->OPCUANames->size(); i++)
        {
            auto v = m_mathExpressionObserver->observe((*singleMode->OPCUANames)[i]);
            if(!v)
                m_rdy = false;
            m_axisValueHandles.push_back(std::move(v));
        }
    }

    auto tool = {m_machineNode->ToolNumberName->get(), m_machineNode->ToolLengthName->get(), m_machineNode->ToolRadiusName->get()};
    for(auto t : tool)
    {
        if(t)
            m_valueIds.push_back(m_client->observeNode(t));
    }
}

void Machine::move(int axis, float value)
{
    if(axis >= m_machineNode->AxisNames->size())
        return;
    auto v = osg::Vec3{*(*m_machineNode->AxisOrientations)[axis], *((*m_machineNode->AxisOrientations)[axis] + 1), *((*m_machineNode->AxisOrientations)[axis] +2) };
    auto osgNode = toOsg((*m_machineNode->AxisNodes)[axis]);
    if(axis <= 2) // ugly hack to find out if an axis is translational
    {
        v *= (value * m_machineNode->OpcUaToVrml->get());
        osgNode->setMatrix(osg::Matrix::translate(v));
    }
    else{
        osgNode->setMatrix(osg::Matrix::rotate(value / 180 *(float)osg::PI, v));
    }
}

bool Machine::arrayMode() const{
    return dynamic_cast<MachineNodeArrayMode *>(m_machineNode) != nullptr;
}

void Machine::setUi(opencover::ui::Menu *menu, opencover::config::File *file)
{
    m_menu = menu;
    m_configFile = file;
}


void Machine::pause(bool state)
{
    if(m_tool) 
        m_tool->value->pause(state);
}

osg::MatrixTransform *Machine::getToolHead() const
{
    return toOsg(m_machineNode->ToolHeadNode->get());
}


void Machine::update()
{
    if(!m_rdy && m_machineNode->allInitialized())
    {
        connectOpcua();
    }
    
    bool haveTool = true;
    if(!m_tool)
    {
        haveTool = addTool();
    }
    if (m_client && m_client->isConnected())
    {
        m_mathExpressionObserver->update();
        updateMachine(haveTool);
    }
}

bool Machine::addTool()
{

    if(strcmp(m_machineNode->VisualizationType->get(), "None") == 0 )
    {
        std::cerr << "missing VisualizationType, make sure this is set in the VRML file to \"Currents\" or \"Oct\"" << std::endl;
        return false;

    }
    auto toolHead = toOsg(m_machineNode->ToolHeadNode->get());
    auto table = toOsg(m_machineNode->TableNode->get());
    if(!toolHead || !table)
    {
        std::cerr << "missing ToolHeadNode or table TableNode, make sure both are set in the VRML file and the corresponding nodes contain some geometry." << std::endl;
        return false;
    }
    ui::Group *machineGroup = new ui::Group(m_menu, m_machineNode->MachineName->get());
    if(strcmp(m_machineNode->VisualizationType->get(), "Currents") == 0 )
    {
        SelfDeletingTool::create(m_tool, std::make_unique<Currents>(machineGroup, *m_configFile, toolHead, table));
        return true;
    }
    if(strcmp(m_machineNode->VisualizationType->get(), "Oct") == 0 )
    {
        SelfDeletingTool::create(m_tool, std::make_unique<Oct>(machineGroup, *m_configFile, toolHead, table));
        dynamic_cast<Oct*>(m_tool->value.get())->setScale(m_machineNode->OpcUaToVrml->get());
        return true;
    }

    return false;
}

bool Machine::updateMachine(bool haveTool)
{
    if(arrayMode())
    {
        auto arrayMode = dynamic_cast<MachineNodeArrayMode *>(m_machineNode);
        auto numUpdates = m_client->numNodeUpdates(arrayMode->OPCUAArrayName->get());
        for (size_t update = 0; update < numUpdates; update++)
        {
            auto v = m_client->getArray<UA_Double>(arrayMode->OPCUAArrayName->get());
            for (size_t i = 0; i < 3; i++)
            {
                move(i, v.data[i] + (*m_machineNode->Offsets)[i]);
            }
            if(haveTool)
                m_tool->value->update(v);
        }
    } else{
        auto singleMode = dynamic_cast<MachineNodeSingleMode *>(m_machineNode);
        if(haveTool)
        {
            m_tool->value->update();
        }

        //machine axis updates
        for (size_t i = 0; i < singleMode->OPCUANames->size(); i++)
        {
            move(i, m_axisValueHandles[i]->value() + (*m_machineNode->Offsets)[i]);
        }
    }
    return true;
}