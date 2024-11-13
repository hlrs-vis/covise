#ifndef COVER_PLUGIN_TOOLMACHINE_VRMLNODE_H
#define COVER_PLUGIN_TOOLMACHINE_VRMLNODE_H

#include <vrml97/vrml/VrmlNodeChildTemplate.h>
#include <vrml97/vrml/VrmlNodeType.h>

class MachineNodeBase : public vrml::VrmlNodeChildTemplate
{
public:
    // static VrmlNode *creator(vrml::VrmlScene *scene);
    MachineNodeBase(vrml::VrmlScene *scene);
    ~MachineNodeBase();
    static vrml::VrmlNodeType *defineType(vrml::VrmlNodeType *t);

    // Set the value of one of the node fields.

    // void setField(const char* fieldName, const VrmlField& fieldValue) override;

    vrml::VrmlSFString *MachineName = nullptr;
    vrml::VrmlSFString *VisualizationType = nullptr;;
    vrml::VrmlSFNode *ToolHeadNode = nullptr;
    vrml::VrmlSFNode *TableNode = nullptr;
    vrml::VrmlMFVec3f *AxisOrientations = nullptr;
    vrml::VrmlMFFloat *Offsets = nullptr;
    vrml::VrmlMFString *AxisNames = nullptr;
    vrml::VrmlSFString *ToolNumberName = nullptr;
    vrml::VrmlSFString *ToolLengthName = nullptr;
    vrml::VrmlSFString *ToolRadiusName = nullptr;
    vrml::VrmlMFNode *AxisNodes = nullptr;
    vrml::VrmlSFFloat *OpcUaToVrml = nullptr;

private:
    size_t m_index = 0; // used to remove the node from the machineNodes list

};
extern std::vector<MachineNodeBase *> machineNodes;

class MachineNodeArrayMode : public MachineNodeBase
{
public:
    static VrmlNode *creator(vrml::VrmlScene *scene);
    static vrml::VrmlNodeType *defineType(vrml::VrmlNodeType *t = nullptr);

    MachineNodeArrayMode(vrml::VrmlScene *scene);
    vrml::VrmlNodeType *nodeType() const override;
    vrml::VrmlNode *cloneMe() const override;
    vrml::VrmlSFString *OPCUAArrayName = nullptr;
    vrml::VrmlMFInt *OPCUAAxisIndicees = nullptr; //array mode expected
};

class MachineNodeSingleMode : public MachineNodeBase
{
public:
    static VrmlNode *creator(vrml::VrmlScene *scene);
    static vrml::VrmlNodeType *defineType(vrml::VrmlNodeType *t = nullptr);

    MachineNodeSingleMode(vrml::VrmlScene *scene);
    vrml::VrmlNodeType *nodeType() const override;
    vrml::VrmlNode *cloneMe() const override;
    vrml::VrmlMFString *OPCUANames = nullptr; //axis names on the opcua server
};

class ToolChangerNode : public vrml::VrmlNodeChildTemplate
{
public:
    static VrmlNode *creator(vrml::VrmlScene *scene);
    static vrml::VrmlNodeType *defineType(vrml::VrmlNodeType *t = nullptr);
    ToolChangerNode(vrml::VrmlScene *scene);
    ~ToolChangerNode();
    vrml::VrmlNodeType *nodeType() const override;
    vrml::VrmlNode *cloneMe() const override;
    vrml::VrmlSFString* arm = nullptr;
    vrml::VrmlSFString* changer = nullptr;
    vrml::VrmlSFString* cover = nullptr;
private:
    size_t m_index = 0; // used to remove the node from the toolChanger list
};

extern std::vector<ToolChangerNode *> toolChangers;

class ToolChangerNodeVrml : public vrml::VrmlNodeChildTemplate
{
public:
    static VrmlNode *creator(vrml::VrmlScene *scene);
    static vrml::VrmlNodeType *defineType(vrml::VrmlNodeType *t = nullptr);
    ToolChangerNodeVrml(vrml::VrmlScene *scene);
    ~ToolChangerNodeVrml();
    vrml::VrmlNodeType *nodeType() const override;
    vrml::VrmlNode *cloneMe() const override;
    vrml::VrmlSFNode *arm = nullptr;
private:
    size_t m_index = 0; // used to remove the node from the toolChanger list
};

extern std::vector<ToolChangerNodeVrml *> toolChangersVrml;



class MachineNode : public vrml::VrmlNodeChildTemplate //dummy to load plugin
{
public:
    static VrmlNode *creator(vrml::VrmlScene *scene);
    static vrml::VrmlNodeType *defineType(vrml::VrmlNodeType *t = nullptr);
    MachineNode(vrml::VrmlScene *scene);

    vrml::VrmlNodeType *nodeType() const override;
    vrml::VrmlNode *cloneMe() const override;

private:
    size_t m_index = 0; // used to remove the node from the toolChanger list
};

#endif // COVER_PLUGIN_TOOLMACHINE_VRMLNODE_H