#include "ToolMachine.h"
#include "opcua.h"
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlNodeTransform.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlSFVec3f.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <util/coExport.h>
#include <vrml97/vrml/VrmlScene.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Menu.h>

#include <stdlib.h>
#include <cover/ui/VectorEditField.h>

using namespace covise;
using namespace opencover;
using namespace vrml;

class MachineNode;
std::vector<MachineNode *> machineNodes;
const std::array<const char*, 5> axisNames{"A", "C", "X", "Y", "Z"};
const std::array<const char*, 5> axisNamesLower{"a", "c", "x", "y", "z"};

static VrmlNode *creator(VrmlScene *scene);

// class PLUGINEXPORT TestYZ : public vrml::VrmlNodeChild
// {
// public:
//     TestYZ(VrmlScene* scene):VrmlNodeChild(scene){}
//     ~TestYZ() = default;
//     VrmlNode *cloneMe() const override
//     {
//         return new TestYZ(*this);
//     }
// };

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
        return t;
    }

    virtual VrmlNodeType *nodeType() const { return defineType(); };

    VrmlNode *cloneMe() const override
    {
        return new MachineNode(*this);
    }

    void move(const osg::Vec3f &position)
    {
        std::cerr << "moving machine" << std::endl;
        auto t = System::the->time();
        eventOut(t, "X", VrmlSFVec3f{-position.x(), 0, 0});
        eventOut(t, "Y", VrmlSFVec3f{0, 0, -position.y()});
        eventOut(t, "Z", VrmlSFVec3f{0, position.z(), 0});
    }

    void move2(int axis, float value)
    {
        auto t = System::the->time();
        if(axis >= 2)
        {
            osg::Vec3f v;
            v[axis -2] = value /1000;
            eventOut(t, axisNames[axis], VrmlSFVec3f{v.x(), v.z(), v.y()});
        }
        else{
            osg::Vec3f v;
            v[axis] = 1;
            eventOut(t, axisNames[axis], VrmlSFRotation{v.x(), v.z(), v.y(), value / 180 *(float)osg::PI});
        }
    }
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
{
    VrmlNamespace::addBuiltIn(MachineNode::defineType());
    auto menu = new ui::Menu("ToolMachine", this);
    m_client.reset(new OpcUaClient("Test", menu, *config()));
    auto g = new opencover::ui::Group(menu, "offsets");
    m_offsets = new opencover::ui::VectorEditField(g, "offset in mm");
    m_client->onConnect([this](){
        
        m_client->registerDouble("ENC2_POS|X", [this](double val)
        {
            for(const auto &m : machineNodes)
                m->move2(2, val + m_offsets->value().x());
        });
        m_client->registerDouble("ENC2_POS|Y", [this](double val)
        {
            for(const auto &m : machineNodes)
                m->move2(3, val + m_offsets->value().y());
        });
        m_client->registerDouble("ENC2_POS|Z", [this](double val)
        {
            for(const auto &m : machineNodes)
                m->move2(4, val + m_offsets->value().z());
        });
        m_client->registerDouble("ENC2_POS|A", [](double val)
        {
            for(const auto &m : machineNodes)
                m->move2(0, val);
        });
        m_client->registerDouble("ENC2_POS|C", [](double val)
        {
            for(const auto &m : machineNodes)
                m->move2(1, val);
        });
    });


}


void ToolMaschinePlugin::key(int type, int keySym, int mod)
{
    if(!type == osgGA::GUIEventAdapter::KEY_Down)
        return;
    std::string key = "unknown";
    if (!(keySym & 0xff00))
    {
        char buf[2] = { static_cast<char>(keySym), '/0' };
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
                m->move2(i, axis);
            }
        }
    }
}

bool ToolMaschinePlugin::update()
{
    m_client->update();
    return true;
}

