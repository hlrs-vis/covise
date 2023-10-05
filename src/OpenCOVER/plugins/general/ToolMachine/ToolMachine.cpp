#include "ToolMachine.h"



#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlMFString.h>
#include <vrml97/vrml/VrmlMFFloat.h>
#include <vrml97/vrml/VrmlMFVec3f.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <plugins/general/Vrml97/ViewerObject.h>

#include <util/coExport.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Menu.h>
#include <cover/VRViewer.h>

#include <stdlib.h>
#include <cover/ui/VectorEditField.h>

#include <OpcUaClient/opcua.h>


using namespace covise;
using namespace opencover;
using namespace vrml;

class MachineNode;
std::vector<MachineNode *> machineNodes;

static VrmlNode *creator(VrmlScene *scene);

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

class MachineNode : public vrml::VrmlNodeChild
{
public:
    static VrmlNode *creator(VrmlScene *scene)
    {
        return new MachineNode(scene);
    }
    MachineNode(VrmlScene *scene) : VrmlNodeChild(scene), m_index(machineNodes.size())
    {

        std::cerr << "vrml Machine node created" << std::endl;
        machineNodes.push_back(this);
    }
    ~MachineNode()
    {
        machineNodes.erase(machineNodes.begin() + m_index);
    }

    static VrmlNodeType *defineType(VrmlNodeType *t = 0)
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
        t->addExposedField("OPCUANames", VrmlField::MFSTRING);
        t->addExposedField("OPCUAAxisIndicees", VrmlField::MFINT32);
        t->addExposedField("AxisOrientations", VrmlField::MFVEC3F);
        t->addExposedField("AxisNodes", VrmlField::MFNODE);
        t->addExposedField("OpcUaToVrml", VrmlField::SFFLOAT); //

        return t;
    }

    // Set the value of one of the node fields.

    void setField(const char* fieldName,
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
            d_client = opcua::connect(d_MachineName.get());
            

        }
        if(!d_rdy && d_MachineName.get() && d_AxisNames.get() && d_ToolHeadNode.get() && d_TableNode.get())
        {
            if(d_OPCUAAxisIndicees.get())
            {
                d_valueIds.push_back(d_client->observeNode(d_OPCUANames[0]));
                d_rdy = true;
            }  else if (d_OPCUANames.get() && d_OPCUANames.size() > 1)
            {
                for (size_t i = 0; i < d_OPCUANames.size(); i++)
                {
                    d_valueIds.push_back(d_client->observeNode(d_OPCUANames[i]));
                }
                d_rdy = true;
            }

        }
    }

    virtual VrmlNodeType *nodeType() const { return defineType(); };

    VrmlNode *cloneMe() const override
    {
        return new MachineNode(*this);
    }

    void move(int axis, float value)
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

    bool arrayMode() const{
        return d_OPCUAAxisIndicees.get() != nullptr;
    }
    VrmlSFString d_MachineName;
    VrmlSFString d_VisualizationType = "None";
    VrmlSFString d_OctOffset;
    VrmlSFNode d_ToolHeadNode;
    VrmlSFNode d_TableNode;
    VrmlMFVec3f d_AxisOrientations;
    VrmlMFFloat d_Offsets;
    VrmlMFString d_AxisNames;
    VrmlMFString d_OPCUANames;
    VrmlMFInt d_OPCUAAxisIndicees;
    VrmlMFNode d_AxisNodes;
    VrmlSFFloat d_OpcUaToVrml = 1;
    bool d_rdy = false;

    opcua::Client *d_client;
    std::vector<opencover::opcua::ObserverHandle> d_valueIds;


private:
    size_t m_index = 0;
};

VrmlNode *creator(VrmlScene *scene)
{
    return new MachineNode(scene);
}



COVERPLUGIN(ToolMaschinePlugin)

ToolMaschinePlugin::ToolMaschinePlugin()
:coVRPlugin(COVER_PLUGIN_NAME)
, ui::Owner("ToolMachinePlugin", cover->ui)
, m_menu(new ui::Menu("ToolMachine", this))
, m_pauseBtn(new ui::Button(m_menu, "pause"))
{
    m_menu->allowRelayout(true);

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
        {
            auto t = m_tools.find(m->d_MachineName.get());
            if(t != m_tools.end()) 
                t->second->value->pause(state);
        }
    });
}

bool ToolMaschinePlugin::addTool(MachineNode *m)
{
    if(strcmp(m->d_VisualizationType.get(), "None") == 0 )
    {
        std::cerr << "missing VisualizationType, make sure this is set in the VRML file to \"Currents\" or \"Oct\"" << std::endl;
        return false;

    }
    auto toolHead = toOsg(m->d_ToolHeadNode.get());
    auto table = toOsg(m->d_TableNode.get());
    if(!toolHead || !table)
    {
        std::cerr << "missing ToolHeadNode or table TableNode, make sure both are set in the VRML file and the corresponding nodes contain some geometry." << std::endl;
        return false;
    }
    ui::Group *machineGroup = new ui::Group(m_menu, m->d_MachineName.get());
    if(strcmp(m->d_VisualizationType.get(), "Currents") == 0 )
    {
        new SelfDeletingTool(m_tools, m->d_MachineName.get(), std::make_unique<Currents>(machineGroup, toolHead, table));
        return true;
    }
    if(strcmp(m->d_VisualizationType.get(), "Oct") == 0 )
    {
        new SelfDeletingTool(m_tools, m->d_MachineName.get(), std::make_unique<Oct>(machineGroup, toolHead, table));
        dynamic_cast<Oct*>(m_tools[m->d_MachineName.get()]->value.get())->setScale(m->d_OpcUaToVrml.get());
        return true;
    }

    return false;
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
    for (const auto& m : machineNodes)
    {
        if(!m->d_rdy)
            return true;
        bool haveTool = true;
        if(m_tools.find(m->d_MachineName.get()) == m_tools.end())
        {
            haveTool = addTool(m);
        }
        auto client = m->d_client;
        if (client && client->isConnected())
        {
            if(m->arrayMode())
            {
                auto numUpdates = client->numNodeUpdates(m->d_OPCUANames[0]);
                for (size_t update = 0; update < numUpdates; update++)
                {
                    auto v = client->getArray<UA_Double>(m->d_OPCUANames[0]);
                    for (size_t i = 0; i < 3; i++)
                    {
                        if(!m_pauseMove && !m_pauseBtn->state())
                            m->move(i, v.data[i] + m->d_Offsets[i]);
                    }
                    if(haveTool)
                        m_tools[m->d_MachineName.get()]->value->update(v);
                }
            } else{
                size_t numUpdates = 1000;
                for (size_t i = 0; i < m->d_OPCUANames.size(); i++)
                {
                    numUpdates = std::min(numUpdates, client->numNodeUpdates(m->d_OPCUANames[i]));
                }
                for (size_t update = 0; update < numUpdates; update++)
                {
                    if(update == numUpdates -1)
                    {
                        for (size_t i = 0; i < m->d_OPCUANames.size(); i++)
                        {
                            auto v = client->getNumericScalar(m->d_OPCUANames[i]);
                            if(!m_pauseMove && !m_pauseBtn->state())
                                m->move(i, v + m->d_Offsets[i]);
                        }
                        if(haveTool)
                            m_tools[m->d_MachineName.get()]->value->update(opcua::MultiDimensionalArray<double>(nullptr));

                    }
                }
            }
            
            
        }
    }

    return true;
}

