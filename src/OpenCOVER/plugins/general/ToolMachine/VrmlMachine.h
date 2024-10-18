#ifndef COVER_PLUGIN_TOOLMACHINE_VRMLNODE_H
#define COVER_PLUGIN_TOOLMACHINE_VRMLNODE_H

#include "LogicInterface.h"

#include <vrml97/vrml/VrmlNodeTemplate.h>
#include <vrml97/vrml/VrmlNodeType.h>

#include <osg/MatrixTransform>
#include <utils/pointer/NullCopyPtr.h>


class MachineNodeBase : public vrml::VrmlNodeTemplate {
public:
    static void initFields(MachineNodeBase *node, vrml::VrmlNodeType *t);
    MachineNodeBase(vrml::VrmlScene *scene, const std::string &name);
    virtual ~MachineNodeBase();


    vrml::VrmlSFString machineName;
    vrml::VrmlSFString visualizationType;
    vrml::VrmlSFNode toolHeadNode;
    vrml::VrmlSFNode tableNode;
    vrml::VrmlMFVec3f axisOrientations;
    vrml::VrmlMFFloat offsets;
    vrml::VrmlMFString axisNames;
    vrml::VrmlSFString toolNumberName;
    vrml::VrmlSFString toolLengthName;
    vrml::VrmlSFString toolRadiusName;
    vrml::VrmlMFNode axisNodes;
    vrml::VrmlSFFloat opcUaToVrml;
    opencover::utils::pointer::NullCopyPtr<LogicInterface> machine;
};

extern std::set<MachineNodeBase *> machineNodes;

class MachineNodeArrayMode : public MachineNodeBase {
public:
    static void initFields(MachineNodeArrayMode *node, vrml::VrmlNodeType *t);
    static const char *name() { return "MachineNodeArrayMode"; }

    MachineNodeArrayMode(vrml::VrmlScene *scene);

    vrml::VrmlSFString opcuaArrayName;
    vrml::VrmlMFInt opcuaAxisIndicees; // array mode expected
};

class MachineNodeSingleMode : public MachineNodeBase {
public:
    static void initFields(MachineNodeSingleMode *node, vrml::VrmlNodeType *t);
    static const char *name() { return "MachineNodeSingleMode"; }

    MachineNodeSingleMode(vrml::VrmlScene *scene);
    
    vrml::VrmlMFString opcuaNames; // axis names on the opcua server
};

class MachineNode : public vrml::VrmlNode { // dummy to load plugin
public:
    static VrmlNode *creator(vrml::VrmlScene *scene);
    static vrml::VrmlNodeType *defineType(vrml::VrmlNodeType *t = nullptr);
    MachineNode(vrml::VrmlScene *scene);

    vrml::VrmlNodeType *nodeType() const override;
    vrml::VrmlNode *cloneMe() const override;
};

#endif // COVER_PLUGIN_TOOLMACHINE_VRMLNODE_H
