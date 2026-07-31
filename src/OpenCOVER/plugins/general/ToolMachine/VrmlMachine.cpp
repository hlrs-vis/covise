#include "VrmlMachine.h"
#include <cassert>

#include <plugins/general/Vrml97/ViewerObject.h>

using namespace vrml;

std::set<MachineNodeBase *> machineNodes;

void MachineNodeBase::initFields(MachineNodeBase *node, VrmlNodeType *t) {
    initFieldsHelper(node, t,
        field("machineName", &MachineNodeBase::machineName),
        field("visualizationType", &MachineNodeBase::visualizationType),
        field("toolHeadNode", &MachineNodeBase::toolHeadNode),
        field("tableNode", &MachineNodeBase::tableNode),
        field("axisOrientations", &MachineNodeBase::axisOrientations),
        field("offsets", &MachineNodeBase::offsets),
        field("axisNames", &MachineNodeBase::axisNames),
        field("toolNumberName", &MachineNodeBase::toolNumberName),
        field("toolLengthName", &MachineNodeBase::toolLengthName),
        field("toolRadiusName", &MachineNodeBase::toolRadiusName),
        field("axisNodes", &MachineNodeBase::axisNodes),
        field("opcUaToVrml", &MachineNodeBase::opcUaToVrml)
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
        field("opcuaAxisIndicees", &MachineNodeArrayMode::opcuaAxisIndicees),
        field("opcuaArrayName", &MachineNodeArrayMode::opcuaArrayName)
    );

}

MachineNodeArrayMode::MachineNodeArrayMode(VrmlScene *scene)
: MachineNodeBase(scene, typeName())
{
}

// single mode


MachineNodeSingleMode::MachineNodeSingleMode(VrmlScene *scene)
: MachineNodeBase(scene, typeName())
{
}

void MachineNodeSingleMode::initFields(MachineNodeSingleMode *node, VrmlNodeType *t) {
    MachineNodeBase::initFields(node, t);
    initFieldsHelper(node, t,
        field("opcuaNames", &MachineNodeSingleMode::opcuaNames)
    );
}

// MachineNode dummy

void MachineNode::initFields(MachineNode *node, VrmlNodeType *t) {
    //do nothing
}

const char *MachineNode::typeName() {
    return "ToolMachine";
}

MachineNode::MachineNode(VrmlScene *scene)
: VrmlNodeChild(scene, typeName())
{}
