#include "ToolMachine.h"

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
#include <open62541/client_config_default.h>
#include <open62541/client_highlevel.h>
#include <open62541/plugin/log_stdout.h>
#include <stdlib.h>


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
            v[axis -2] = value;
            eventOut(t, axisNames[axis], VrmlSFVec3f{v.x(), v.z(), v.y()});
        }
        else{
            osg::Vec3f v;
            v[axis] = 1;
            eventOut(t, axisNames[axis], VrmlSFRotation{v.x(), v.z(), v.y(), value});
        }
    }
    void aAxisYOffset(float val)
    {
        auto t = System::the->time();
        eventOut(t, "aAxisYOffsetPos",VrmlSFVec3f(0, val, 0));
        eventOut(t, "aAxisYOffsetNeg",VrmlSFVec3f(0, -1 *val, 0));
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
, ui::Owner("ToolMaschinePlugin", cover->ui)
{
    VrmlNamespace::addBuiltIn(MachineNode::defineType());
    auto menu = new ui::Menu("ToolMaschine", this);
    auto aAxisYOffset = new ui::Slider(menu, "aAxisYOffset");
    aAxisYOffset->setBounds(-2, 2);
    aAxisYOffset->setCallback([](double val, bool rel){
        for(auto &m : machineNodes)
        {
            m->aAxisYOffset(val);
        }
    });



/* Create the server and set its config */
    UA_Client *client = UA_Client_new();
    UA_ClientConfig *cc = UA_Client_getConfig(client);

    UA_ClientConfig_setDefault(cc);
    cc->securityMode = UA_MESSAGESECURITYMODE_NONE;
    UA_ByteString_clear(&cc->securityPolicyUri);
    cc->securityPolicyUri = UA_STRING_NULL;


    /* The application URI must be the same as the one in the certificate.
     * The script for creating a self-created certificate generates a certificate
     * with the Uri specified below.*/
    UA_ApplicationDescription_clear(&cc->clientDescription);
    cc->clientDescription.applicationUri = UA_STRING_ALLOC("urn:open62541.server.application");
    cc->clientDescription.applicationType = UA_APPLICATIONTYPE_CLIENT;

    /* Connect to the server */
    UA_StatusCode retval = UA_STATUSCODE_GOOD;
    UA_ClientConfig_setAuthenticationUsername(cc, "Woessner_hlrs", "34$Wa99#*");
    retval = UA_Client_connect(client, "opc.tcp://141.58.132.65:48010");
    /* Alternative */
    //retval = UA_Client_connectUsername(client, serverurl, username, password);

    if(retval != UA_STATUSCODE_GOOD) {
        UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "Could not connect");
        UA_Client_delete(client);
        return;
    }

    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "Connected!");

    /* Read the server-time */
    UA_Variant value;
    UA_Variant_init(&value);
    UA_Client_readValueAttribute(client,
              UA_NODEID_NUMERIC(0, UA_NS0ID_SERVER_SERVERSTATUS_CURRENTTIME),
              &value);
    if(UA_Variant_hasScalarType(&value, &UA_TYPES[UA_TYPES_DATETIME])) {
        UA_DateTimeStruct dts = UA_DateTime_toStruct(*(UA_DateTime *)value.data);
        UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND,
                    "The server date is: %02u-%02u-%04u %02u:%02u:%02u.%03u",
                    dts.day, dts.month, dts.year, dts.hour, dts.min, dts.sec, dts.milliSec);
    }
    UA_Variant_clear(&value);

    /* Clean up */
    UA_Client_disconnect(client);
    UA_Client_delete(client);


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
                m->move2(i, axis);
            }
        }
    }
}

