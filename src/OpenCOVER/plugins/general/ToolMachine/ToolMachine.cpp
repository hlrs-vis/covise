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
        t->addExposedField("ToolHeadNode", VrmlField::SFNODE);
        t->addExposedField("TableNode", VrmlField::SFNODE);
        t->addExposedField("Offsets", VrmlField::MFFLOAT);
        t->addExposedField("AxisNames", VrmlField::MFSTRING);
        t->addExposedField("OPCUANames", VrmlField::MFSTRING);
        t->addExposedField("AxisOrientations", VrmlField::MFVEC3F);
        t->addExposedField("AxisNodes", VrmlField::MFNODE);

        return t;
    }

    // Set the value of one of the node fields.

    void setField(const char* fieldName,
        const VrmlField& fieldValue)
    {
        if
            TRY_FIELD(MachineName, SFString)
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
            TRY_FIELD(AxisNodes, MFNode)
        else
            VrmlNodeChild::setField(fieldName, fieldValue);
        if (strcmp(fieldName, "MachineName") == 0)
        {
            //connect to the specified machine through OPC-UA
            opcua::connect(d_MachineName.get());

        }
        if(d_MachineName.get() && d_AxisNames.get() && d_ToolHeadNode.get() && d_TableNode.get())
            d_rdy = true;
    }

    virtual VrmlNodeType *nodeType() const { return defineType(); };

    VrmlNode *cloneMe() const override
    {
        return new MachineNode(*this);
    }

    void move(int axis, float value)
    {
        auto v = osg::Vec3{*d_AxisOrientations[axis], *(d_AxisOrientations[axis] + 1), *(d_AxisOrientations[axis] +2) };
        auto osgNode = toOsg(d_AxisNodes[axis]);
        if(axis <= 2) // ugly hack to find out if an axis is translational
        {
            v *= (value /1000.0);
            osgNode->setMatrix(osg::Matrix::translate(v));
        }
        else{
            osgNode->setMatrix(osg::Matrix::rotate(value / 180 *(float)osg::PI, v));
        }
    }

    VrmlSFString d_MachineName;
    VrmlSFNode d_ToolHeadNode;
    VrmlSFNode d_TableNode;
    VrmlMFVec3f d_AxisOrientations;
    VrmlMFFloat d_Offsets;
    VrmlMFString d_AxisNames;
    VrmlMFString d_OPCUANames;
    VrmlMFNode d_AxisNodes;
    bool d_rdy = false;

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
{
    m_menu->allowRelayout(true);

    opcua::addOnClientConnectedCallback([this]()
    {
        //auto availableFields = opcua::getClient()->availableNumericalScalars();
    });
    

    opcua::addOnClientDisconnectedCallback([this]()
    {
    });
    
    
    VrmlNamespace::addBuiltIn(MachineNode::defineType());

    config()->setSaveOnExit(true);
    

    // m_offsets = new opencover::ui::VectorEditField(menu, "offsetInMM");
    // m_offsets->setValue(osg::Vec3(-406.401596,324.97962,280.54943));

}

void ToolMaschinePlugin::addCurrent(MachineNode *m)
{
    auto toolHead = toOsg(m->d_ToolHeadNode.get());
    auto table = toOsg(m->d_TableNode.get());
    if(!toolHead || !table)
        return;
    ui::Group *machineGroup = new ui::Group(m_menu, m->d_MachineName.get());
    auto &currents = m_currents.insert(std::make_pair(m->d_MachineName.get(), Currents(machineGroup, toolHead, table))).first->second;
    // std::array<ui::Slider *, 5> sliders;
    // for (size_t i = 0; i < 5; i++)
    // {
    //     sliders[i] = new ui::Slider(machineGroup, *m->d_AxisNames[i] + std::string("slider"));
    // }

    // for (size_t i = 0; i < 5; i++)
    // {
    //     sliders[i]->setBounds(-1, 1);
    //     sliders[i]->setCallback([this, &currents, sliders](ui::Slider::ValueType val, bool rel){
    //         std::array<double, 5> sliderVals;
    //         for (size_t i = 0; i < 5; i++)
    //         {
    //             sliderVals[i] = sliders[i]->value();
    //         }
    //         currents.setOffset(sliderVals);
    //     });
    // }
}

void ToolMaschinePlugin::key(int type, int keySym, int mod)
{
    std::string key = "unknown";
    if (!(keySym & 0xff00))
    {
        char buf[2] = { static_cast<char>(keySym), '\0' };
        key = buf;
    }
    float speed = 3;
    std::cerr << "Key input  " << key << std::endl;

    for (auto m : machineNodes)
    {
        for (size_t i = 0; i < m->d_AxisNames.size(); i++)
        {
            auto& axis = m_axisPositions[i];
            if (key[0] == std::tolower(m->d_AxisNames[i][0]))
            {
                if (mod == 3)
                {
                    std::cerr << "decreasing " << m->d_AxisNames[i] << std::endl;
                    speed *= -1;
                }
                else {
                    std::cerr << "increasing " << m->d_AxisNames[i] << std::endl;
                }
                axis += speed;
                // m->move(v);
                m->move(i, axis);
            }
        }
    }
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
        if(m_currents.find(m->d_MachineName.get()) == m_currents.end())
            addCurrent(m);
        auto client = opcua::getClient(m->d_MachineName.get()); // get the client with this name
        if (client && client->isConnected())
        {
            for (size_t i = 0; i < m->d_OPCUANames.size(); i++)
            {
                m->move(i, client->readNumericValue(m->d_OPCUANames[i]) + m->d_Offsets[i]);
            }
        }
    }
    for(auto & c : m_currents)
        c.second.update();
    return true;
}

