#include "VrmlMachine.h"
#include <cassert>

#include <plugins/general/Vrml97/ViewerObject.h>

using namespace vrml;

std::set<MachineNodeBase *> machineNodes;

void MachineNodeBase::initFields(MachineNodeBase *node, VrmlNodeType *t) {
    initFieldsHelper(node, t,
        field("machineName", node->machineName),
        field("visualizationType", node->visualizationType),
        field("toolHeadNode", node->toolHeadNode),
        field("tableNode", node->tableNode),
        field("axisOrientations", node->axisOrientations),
        field("offsets", node->offsets),
        field("axisNames", node->axisNames),
        field("toolNumberName", node->toolNumberName),
        field("toolLengthName", node->toolLengthName),
        field("toolRadiusName", node->toolRadiusName),
        field("axisNodes", node->axisNodes),
        field("opcUaToVrml", node->opcUaToVrml)
    );
}

MachineNodeBase::MachineNodeBase(vrml::VrmlScene *scene, const std::string &name)
: VrmlNodeTemplate(scene, name)
{
    machineNodes.emplace(this);
}

MachineNodeBase::~MachineNodeBase()
{
    machineNodes.erase(this);
}

// array mode
void MachineNodeArrayMode::initFields(MachineNodeArrayMode *node, VrmlNodeType *t) {
    
    MachineNodeBase::initFields(node, t);
    initFieldsHelper(node, t,
        field("opcuaAxisIndicees", node->opcuaAxisIndicees),
        field("opcuaArrayName", node->opcuaArrayName)
    );

}

MachineNodeArrayMode::MachineNodeArrayMode(VrmlScene *scene)
: MachineNodeBase(scene, name())
{
    initFields(this, nullptr);
}

// single mode


MachineNodeSingleMode::MachineNodeSingleMode(VrmlScene *scene)
: MachineNodeBase(scene, name())
{
    initFields(this, nullptr);
}

void MachineNodeSingleMode::initFields(MachineNodeSingleMode *node, VrmlNodeType *t) {
    MachineNodeBase::initFields(node, t);
    initFieldsHelper(node, t,
        field("opcuaNames", node->opcuaNames)
    );
}

// MachineNode dummy

MachineNode::MachineNode(VrmlScene *scene)
: VrmlNode(scene)
{}

VrmlNode *MachineNode::creator(VrmlScene *scene)
{
    return new MachineNode(scene);
}

VrmlNodeType *MachineNode::nodeType() const { return defineType(); };

VrmlNode *MachineNode::cloneMe() const 
{
    return new MachineNode(*this);
}

VrmlNodeType *MachineNode::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;
    std::cerr << "defining ToolMachine type " << std::endl;
    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("ToolMachine", creator);
    }
    return t;
}
