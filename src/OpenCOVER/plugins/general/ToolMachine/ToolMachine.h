
#include "Tool.h"

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
#include <OpcUaClient/opcua.h>
#include <cover/ui/Menu.h>

enum UpdateMode
{
    All,
    AllOncePerFrame,
    UpdatedOncePerFrame
};
class MachineNode : public vrml::VrmlNodeChild
{
public:
    static VrmlNode *creator(VrmlScene *scene);
    MachineNode(VrmlScene *scene);
    ~MachineNode();

    static VrmlNodeType *defineType(VrmlNodeType *t = 0);

    template<typename T>
    void triggerEvent(const char* name, T value)
    {
        auto t = System::the->time();
        eventOut(t, name, value);
    }
    // Set the value of one of the node fields.

    void setField(const char* fieldName, const VrmlField& fieldValue);

    virtual VrmlNodeType *nodeType() const;

    VrmlNode *cloneMe() const override;
    void move(int axis, float value);
    bool arrayMode() const;
    void update(UpdateMode updateMode);
    void setUi(opencover::ui::Menu *menu, opencover::config::File *file);
    void pause(bool state);

    VrmlSFString d_MachineName;
    VrmlSFString d_VisualizationType = "None";
    VrmlSFString d_OctOffset;
    VrmlSFNode d_ToolHeadNode;
    VrmlSFNode d_TableNode;
    VrmlMFVec3f d_AxisOrientations;
    VrmlMFFloat d_Offsets;
    VrmlMFString d_AxisNames;
    VrmlSFString d_ToolNumberName;
    VrmlSFString d_ToolLengthName;
    VrmlSFString d_ToolRadiusName;
    VrmlMFString d_OPCUANames; //axis names on the opcua server
    VrmlMFInt d_OPCUAAxisIndicees;
    VrmlMFNode d_AxisNodes;
    VrmlSFFloat d_OpcUaToVrml = 1;
private:
    bool d_rdy = false;

    opcua::Client *m_client;
    std::vector<opencover::opcua::ObserverHandle> d_valueIds;
    size_t m_index = 0;
    //VrmlNodeChild must be copyable
    std::shared_ptr<std::unique_ptr<SelfDeletingTool>> m_tool;
    opencover::ui::Menu *m_menu;
    opencover::config::File *m_configFile;

    bool addTool();
    bool updateMachine(bool haveTool, UpdateMode updateMode);
};

extern std::vector<MachineNode *> machineNodes;
