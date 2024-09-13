#include "ToolMachine.h"
#include "Currents.h"
#include "Oct.h"

#include <vrml97/vrml/VrmlScene.h>


using namespace covise;
using namespace opencover;
using namespace vrml;

std::vector<MachineNode *> machineNodes;


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


VrmlNode *MachineNode::creator(VrmlScene *scene)
{
    return new MachineNode(scene);
}
MachineNode::MachineNode(VrmlScene *scene)
: VrmlNodeChild(scene), m_index(machineNodes.size())
, m_tool(std::make_shared<std::unique_ptr<SelfDeletingTool>>())

{

    std::cerr << "vrml Machine node created" << std::endl;
    machineNodes.push_back(this);
}

MachineNode::~MachineNode()
{
    machineNodes.erase(machineNodes.begin() + m_index);
}

VrmlNodeType *MachineNode::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("ToolMachine", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class

    t->addExposedField("MachineName", VrmlField::SFSTRING);
    t->addExposedField("VisualizationType", VrmlField::SFSTRING); //None, Currents, Oct
    t->addExposedField("OctOffset", VrmlField::SFSTRING); 
    t->addExposedField("ToolHeadNode", VrmlField::SFNODE);
    t->addExposedField("TableNode", VrmlField::SFNODE);
    t->addExposedField("Offsets", VrmlField::MFFLOAT);
    t->addExposedField("AxisNames", VrmlField::MFSTRING);
    t->addExposedField("ToolNumberName", VrmlField::SFSTRING);
    t->addExposedField("ToolLengthName", VrmlField::SFSTRING);
    t->addExposedField("ToolRadiusName", VrmlField::SFSTRING);
    t->addExposedField("OPCUANames", VrmlField::MFSTRING);
    t->addExposedField("OPCUAAxisIndicees", VrmlField::MFINT32);
    t->addExposedField("AxisOrientations", VrmlField::MFVEC3F);
    t->addExposedField("AxisNodes", VrmlField::MFNODE);
    t->addExposedField("OpcUaToVrml", VrmlField::SFFLOAT); //

    t->addEventOut("ToolNumber", VrmlField::SFINT32);
    t->addEventOut("ToolLength", VrmlField::SFFLOAT);
    t->addEventOut("ToolRadius", VrmlField::SFFLOAT);
    return t;
}

    // Set the value of one of the node fields.

void MachineNode::setField(const char* fieldName,
    const VrmlField& fieldValue)
{
    if
        TRY_FIELD(MachineName, SFString)
    else if
        TRY_FIELD(VisualizationType, SFString)
    else if
        TRY_FIELD(OctOffset, SFString)
    else if
        TRY_FIELD(ToolHeadNode, SFNode)
    else if
        TRY_FIELD(TableNode, SFNode)
    else if
        TRY_FIELD(AxisOrientations, MFVec3f)
    else if
        TRY_FIELD(Offsets, MFFloat)
    else if
        TRY_FIELD(AxisNames, MFString)
    else if
        TRY_FIELD(ToolNumberName, SFString)
    else if
        TRY_FIELD(ToolLengthName, SFString)
    else if
        TRY_FIELD(ToolRadiusName, SFString)
    else if
        TRY_FIELD(OPCUANames, MFString)
    else if
        TRY_FIELD(OPCUAAxisIndicees, MFInt)
    else if
        TRY_FIELD(AxisNodes, MFNode)
    else if
        TRY_FIELD(OpcUaToVrml, SFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
    if (strcmp(fieldName, "MachineName") == 0)
    {
        //connect to the specified machine through OPC-UA
        m_client = opcua::connect(d_MachineName.get());
        

    }
    if(!d_rdy && d_MachineName.get() && d_AxisNames.get() && d_ToolHeadNode.get() && d_TableNode.get())
    {
        if(d_OPCUAAxisIndicees.get())
        {
            d_valueIds.push_back(m_client->observeNode(d_OPCUANames[0]));
            d_rdy = true;
        }  else if (d_OPCUANames.get() && d_OPCUANames.size() > 1)
        {
            for (size_t i = 0; i < d_OPCUANames.size(); i++)
            {
                d_valueIds.push_back(m_client->observeNode(d_OPCUANames[i]));
            }
            d_rdy = true;
        }

    }
    auto tool = {d_ToolNumberName.get(), d_ToolLengthName.get(), d_ToolRadiusName.get()};
    for(auto t : tool)
    {
        if(t)
            d_valueIds.push_back(m_client->observeNode(t));
    }
}

VrmlNodeType *MachineNode::nodeType() const { return defineType(); };

VrmlNode *MachineNode::cloneMe() const 
{
    return new MachineNode(*this);
}

void MachineNode::move(int axis, float value)
{
    if(axis >= d_AxisNames.size())
        return;
    auto v = osg::Vec3{*d_AxisOrientations[axis], *(d_AxisOrientations[axis] + 1), *(d_AxisOrientations[axis] +2) };
    auto osgNode = toOsg(d_AxisNodes[axis]);
    if(axis <= 2) // ugly hack to find out if an axis is translational
    {
        v *= (value * d_OpcUaToVrml.get());
        osgNode->setMatrix(osg::Matrix::translate(v));
    }
    else{
        osgNode->setMatrix(osg::Matrix::rotate(value / 180 *(float)osg::PI, v));
    }
}

bool MachineNode::arrayMode() const{
    return d_OPCUAAxisIndicees.get() != nullptr;
}


VrmlNode *creator(VrmlScene *scene)
{
    return new MachineNode(scene);
}

void MachineNode::setUi(opencover::ui::Menu *menu, opencover::config::File *file)
{
    m_menu = menu;
    m_configFile = file;
}


void MachineNode::pause(bool state)
{
    if(*m_tool) 
        m_tool.get()->get()->value->pause(state);
}


void MachineNode::update(UpdateMode updateMode)
{
    if(!d_rdy)
        return;
    
    bool haveTool = true;
    if(!*m_tool)
    {
        haveTool = addTool();
    }
    if (m_client && m_client->isConnected())
    {
        updateMachine(haveTool, updateMode);
    }
}

bool MachineNode::addTool()
{

    if(strcmp(d_VisualizationType.get(), "None") == 0 )
    {
        std::cerr << "missing VisualizationType, make sure this is set in the VRML file to \"Currents\" or \"Oct\"" << std::endl;
        return false;

    }
    auto toolHead = toOsg(d_ToolHeadNode.get());
    auto table = toOsg(d_TableNode.get());
    if(!toolHead || !table)
    {
        std::cerr << "missing ToolHeadNode or table TableNode, make sure both are set in the VRML file and the corresponding nodes contain some geometry." << std::endl;
        return false;
    }
    ui::Group *machineGroup = new ui::Group(m_menu, d_MachineName.get());
    auto &ptr = *m_tool.get();
    if(strcmp(d_VisualizationType.get(), "Currents") == 0 )
    {
        SelfDeletingTool::create(ptr, std::make_unique<Currents>(machineGroup, *m_configFile, toolHead, table));
        return true;
    }
    if(strcmp(d_VisualizationType.get(), "Oct") == 0 )
    {
        SelfDeletingTool::create(ptr, std::make_unique<Oct>(machineGroup, *m_configFile, toolHead, table));
        dynamic_cast<Oct*>(m_tool.get()->get()->value.get())->setScale(d_OpcUaToVrml.get());
        return true;
    }

    return false;
}

bool MachineNode::updateMachine(bool haveTool, UpdateMode updateMode)
{
    if(arrayMode())
    {
        auto numUpdates = m_client->numNodeUpdates(d_OPCUANames[0]);
        for (size_t update = 0; update < numUpdates; update++)
        {
            auto v = m_client->getArray<UA_Double>(d_OPCUANames[0]);
            for (size_t i = 0; i < 3; i++)
            {
                move(i, v.data[i] + d_Offsets[i]);
            }
            if(haveTool)
                m_tool.get()->get()->value->update(v);
        }
    } else{
        struct Update{
            double value;
            std::function<void(double)> func;
            std::string name;
        };
        //read all updates first, then apply them in order
        std::map<UA_DateTime, Update> updates; 

        //vrml specific updates
        std::vector<UpdateValues> toolUpdateValues{
            {d_ToolNumberName.get(), [this](double value){
                triggerEvent<VrmlSFInt>("ToolNumber", value);
            }},
            {d_ToolLengthName.get(), [this](double value){
                triggerEvent<VrmlSFFloat>("ToolLength", value);
            }},
            {d_ToolRadiusName.get(), [this](double value){
                triggerEvent<VrmlSFFloat>("ToolRadius", value);
            }}  
        };
        if(haveTool)
        {
            //get the tool specific update functions 
            auto &tua = m_tool.get()->get()->value->getUpdateValues();
            for(auto &t : tua)
                toolUpdateValues.push_back(t);
                
        }
        for(const auto &update : toolUpdateValues)
        {
            auto numUpdates = m_client->numNodeUpdates(update.name);
            if(numUpdates == 0 && updateMode == UpdateMode::AllOncePerFrame)
                numUpdates = 1;
            for (size_t u = 0; u < numUpdates; u++)
            {
                UA_DateTime timestamp;
                auto v = m_client->getNumericScalar(update.name, &timestamp);
                updates[timestamp] = {v, update.func, update.name};
            }
        }
        //machine axis updates
        for (size_t i = 0; i < d_OPCUANames.size(); i++)
        {
            auto numUpdates = m_client->numNodeUpdates(d_OPCUANames[i]);
            if(numUpdates == 0 && updateMode == UpdateMode::AllOncePerFrame)
                numUpdates = 1;

            for (size_t u = 0; u < numUpdates; u++)
            {
                UA_DateTime timestamp;
                auto v = m_client->getNumericScalar(d_OPCUANames[i], &timestamp);
                updates[timestamp] = {v, [this, i](double value){//potentially overwrite the value with the same timestamp, could be bad
                    move(i, value + d_Offsets[i]);
                }, d_OPCUANames[i]}; 
            }
        }
        //machine tool updates

        //execute updates in order
        if(updateMode == UpdateMode::All)
        {
            for(const auto &update : updates)
            {
                update.second.func(update.second.value);
            }
        } else{
            std::set<std::string> items;
            for(auto it = updates.rbegin(); it != updates.rend(); ++it)
            {
                if(items.insert(it->second.name).second)
                {
                    it->second.func(it->second.value);
                }
            }
        }
        if(haveTool)
            m_tool.get()->get()->value->frameOver();
    }
    return true;
}