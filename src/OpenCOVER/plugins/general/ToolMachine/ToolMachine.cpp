#include "ToolMachine.h"

#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlMFString.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <util/coExport.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Menu.h>

#include <stdlib.h>
#include <cover/ui/VectorEditField.h>

#include <OpcUaClient/opcua.h>

using namespace covise;
using namespace opencover;
using namespace vrml;

class MachineNode;
std::vector<MachineNode *> machineNodes;
const std::array<const char*, 5> axisNames{"A", "C", "X", "Y", "Z"};
const std::array<const char*, 5> axisNamesLower{"a", "c", "x", "y", "z"};

static VrmlNode *creator(VrmlScene *scene);

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
        for (size_t i = 0; i < 2; i++)
        {
            t->addEventOut(axisNames[i], VrmlField::SFROTATION);
        }
        for (size_t i = 2; i < axisNames.size(); i++)
        {
            t->addEventOut(axisNames[i], VrmlField::SFVEC3F);
        }
         t->addEventOut("aAxisYOffsetPos", VrmlField::SFVEC3F);
         t->addEventOut("aAxisYOffsetNeg", VrmlField::SFVEC3F);
         t->addExposedField("MachineName", VrmlField::SFSTRING);
         t->addExposedField("XAxis", VrmlField::SFVEC3F);
         t->addExposedField("YAxis", VrmlField::SFVEC3F);
         t->addExposedField("ZAxis", VrmlField::SFVEC3F);
         t->addExposedField("AAxis", VrmlField::SFVEC3F);
         t->addExposedField("BAxis", VrmlField::SFVEC3F);
         t->addExposedField("CAxis", VrmlField::SFVEC3F);
         t->addExposedField("AxisNames", VrmlField::MFSTRING);
        return t;
    }

    // Set the value of one of the node fields.

    void setField(const char* fieldName,
        const VrmlField& fieldValue)
    {
        if
            TRY_FIELD(MachineName, SFString)
        else if
            TRY_FIELD(XAxis, SFVec3f)
        else if
            TRY_FIELD(YAxis, SFVec3f)
        else if
            TRY_FIELD(ZAxis, SFVec3f)
        else if
            TRY_FIELD(AAxis, SFVec3f)
        else if
            TRY_FIELD(BAxis, SFVec3f)
        else if
            TRY_FIELD(CAxis, SFVec3f)
        else if
            TRY_FIELD(AxisNames, MFString)
        else
            VrmlNodeChild::setField(fieldName, fieldValue);
        if (strcmp(fieldName, "MachineName") == 0)
        {
            //connect to the specified machine through OPC-UA
        }
        else
        {

        }
    }

    virtual VrmlNodeType *nodeType() const { return defineType(); };

    VrmlNode *cloneMe() const override
    {
        return new MachineNode(*this);
    }

    void move(int axis, float value)
    {
        auto t = System::the->time();
        VrmlSFVec3f v;
        if (strcmp(d_AxisNames[axis], "X") == 0)
        {
            v = d_XAxis;
        }
        else if (strcmp(d_AxisNames[axis], "Y") == 0)
        {
            v = d_YAxis;
        }
        else if (strcmp(d_AxisNames[axis], "Z") == 0)
        {
            v = d_ZAxis;
        }
        else if (strcmp(d_AxisNames[axis], "A") == 0)
        {
            v = d_AAxis;
        }
        else if (strcmp(d_AxisNames[axis], "B") == 0)
        {
            v = d_BAxis;
        }
        else if (strcmp(d_AxisNames[axis], "C") == 0)
        {
            v = d_CAxis;
        }
        if(axis >= 2) // ugly hack to find out if an axis is translational
        {
            v.multiply(value /1000.0);
            eventOut(t, d_AxisNames[axis], v);
        }
        else{
            eventOut(t, d_AxisNames[axis], VrmlSFRotation{v.x(), v.y(), v.z(), value / 180 *(float)osg::PI});
        }
    }
    VrmlSFString d_MachineName;
    VrmlSFVec3f d_XAxis;
    VrmlSFVec3f d_YAxis;
    VrmlSFVec3f d_ZAxis;
    VrmlSFVec3f d_AAxis;
    VrmlSFVec3f d_BAxis;
    VrmlSFVec3f d_CAxis;
    VrmlMFString d_AxisNames;
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
, m_server(std::make_unique<ui::EditFieldConfigValue>(m_menu, "server", "", *config(), "", config::Flag::PerModel))
{
    
    m_menu->allowRelayout(true);

    for (size_t i = 0; i < 5; i++)
        m_axisNames[i] = std::make_unique<ui::SelectionListConfigValue>(m_menu, axisNames[i] + std::string("_axis"), 0, *config(), "axles", config::Flag::PerModel);
    for (size_t i = 0; i < 5; i++)
        m_offsets[i] = std::make_unique<ui::EditFieldConfigValue>(m_menu, axisNames[i] + std::string("_offset"), "", *config(), "offsets", config::Flag::PerModel);

    opcua::addOnClientConnectedCallback([this]()
    {
        auto availableFields = opcua::getClient()->availableNumericalScalars();
        for (size_t i = 0; i < 5; i++)
        {
            m_axisNames[i]->ui()->setList(availableFields);
            m_axisNames[i]->ui()->select(m_axisNames[i]->getValue());
        }
    });


    m_server->setUpdater([this]()
    {
        opcua::connect(m_server->getValue());
    });

    opcua::addOnClientDisconnectedCallback([this]()
    {
        for (size_t i = 0; i < 5; i++)
        {
            m_axisNames[i]->ui()->setList(std::vector<std::string>());
        }
    });
    
    
    VrmlNamespace::addBuiltIn(MachineNode::defineType());

    config()->setSaveOnExit(true);

    std::array<ui::Slider *, 5> sliders;
    for (size_t i = 0; i < 5; i++)
    {
        sliders[i] = new ui::Slider(m_menu, axisNames[i] + std::string("slider"));
    }
    for (size_t i = 0; i < 5; i++)
    {
        sliders[i]->setBounds(-1, 1);
        sliders[i]->setCallback([this, sliders](ui::Slider::ValueType val, bool rel){
            std::array<double, 5> sliderVals;
            for (size_t i = 0; i < 5; i++)
            {
                sliderVals[i] = sliders[i]->value();
            }
            m_currents.setOffset(sliderVals);
        });
    }

    // m_offsets = new opencover::ui::VectorEditField(menu, "offsetInMM");
    // m_offsets->setValue(osg::Vec3(-406.401596,324.97962,280.54943));

    
}


void ToolMaschinePlugin::key(int type, int keySym, int mod)
{
    if(!type == osgGA::GUIEventAdapter::KEY_Down)
        return;
    std::string key = "unknown";
    if (!(keySym & 0xff00))
    {
        char buf[2] = { static_cast<char>(keySym), '\0' };
        key = buf;
    }
    float speed = 0.5;
    std::cerr << "Key input  " << key << std::endl;
    
    for (size_t i = 0; i < axisNames.size(); i++)
    {
        auto &axis = m_axisPositions[i];
        if(key == axisNamesLower[i])
        {
            if(mod == 3)
            {
                std::cerr << "decreasing " << axisNames[i] << std::endl;
                speed *= -1;
            } else{
                std::cerr << "increasing " << axisNames[i] << std::endl;
            }
            axis += speed;
            for(auto &m : machineNodes)
            {
                // m->move(v);
                m->move(i, axis);
            }
            m_currents.update(m_axisPositions, m_axisPositions);

        }
    }
}

bool ToolMaschinePlugin::update()
{
    for (const auto& m : machineNodes)
    {
        
        auto client = opcua::getClient(/*m->d_MachineName.get()*/); // get the client with this name
        if (!client || !client->isConnected())
            return true;
        std::array<double, 5> axisValues;

        for (size_t i = 0; i < 5; i++)
            axisValues[i] = client->readNumericValue(m_axisNames[i]->ui()->selectedItem()) + m_offsets[i]->ui()->number();
        for (size_t i = 0; i < 5; i++)
            m->move(i, axisValues[i]);

        m_currents.update(axisValues, axisValues);
    }
    return true;
}

