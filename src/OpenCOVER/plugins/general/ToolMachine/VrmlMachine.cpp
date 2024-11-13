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
: VrmlNodeChild(scene, name)
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
}

// single mode


MachineNodeSingleMode::MachineNodeSingleMode(VrmlScene *scene)
: MachineNodeBase(scene, name())
{
}

void MachineNodeSingleMode::initFields(MachineNodeSingleMode *node, VrmlNodeType *t) {
    MachineNodeBase::initFields(node, t);
    initFieldsHelper(node, t,
        field("opcuaNames", node->opcuaNames)
    );
}

// MachineNode dummy

void MachineNode::initFields(MachineNode *node, VrmlNodeType *t) {
    //do nothing
}

const char *MachineNode::name() {
    return "ToolMachine";
}

MachineNode::MachineNode(VrmlScene *scene)
: VrmlNodeChild(scene, name())
{}
