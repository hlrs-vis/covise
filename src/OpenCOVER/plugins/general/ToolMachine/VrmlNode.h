#ifndef COVER_PLUGIN_TOOLMACHINE_VRMLNODE_H
#define COVER_PLUGIN_TOOLMACHINE_VRMLNODE_H

#include <vrml97/vrml/VrmlNodeChildTemplate.h>
#include <vrml97/vrml/VrmlNodeType.h>

#include <osg/MatrixTransform>


class LogicInterface{
public:
    virtual void update() = 0;
    virtual ~LogicInterface() = default;
};

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
    std::shared_ptr<LogicInterface> machine;
private:
    size_t m_index = 0; // used to remove the node from the machineNodes list

};
extern std::set<MachineNodeBase *> machineNodes;

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
    vrml::VrmlSFNode *toolHead = nullptr;
    std::shared_ptr<LogicInterface> toolChanger;
private:
    size_t m_index = 0; // used to remove the node from the toolChanger list
};

extern std::set<ToolChangerNode *> toolChangers;

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

osg::MatrixTransform *toOsg(vrml::VrmlNode *node);

#endif // COVER_PLUGIN_TOOLMACHINE_VRMLNODE_H