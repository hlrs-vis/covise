#include "VrmlNode.h"
#include <cassert>

using namespace vrml;

std::vector<MachineNodeBase *> machineNodes;
std::vector<ToolChangerNode *> toolChangers;

// template <typename... Ts>
// auto make_variant_vector(Ts&&... args) {
//     using VariantType = std::variant<std::decay_t<Ts>...>;
//     return std::vector<VariantType>{std::forward<Ts>(args)...};
// }

// void initFieldsHelper(MachineNodeBase *node, VrmlNodeType *t)
// {
//     auto test = make_variant_vector(&node->MachineName, &node->VisualizationType, &node->ToolHeadNode, &node->TableNode, &node->AxisOrientations, &node->Offsets, &node->AxisNames, &node->ToolNumberName, &node->ToolLengthName, &node->ToolRadiusName, &node->AxisNodes, &node->OpcUaToVrml);
//     std::vector<std::string> nodeNames = {"MachineName", "VisualizationType", "ToolHeadNode", "TableNode", "AxisOrientations", "Offsets", "AxisNames", "ToolNumberName", "ToolLengthName", "ToolRadiusName", "AxisNodes", "OpcUaToVrml"};
    
//     if(node)
//     {
//         for (size_t i = 0; i < test.size(); i++)
//         {
//             auto &var = test[i];
//             auto &name = nodeNames[i];
//             std::visit([node, name](auto&& arg) {
//                 arg = node->registerField<std::decay_t<decltype(*arg)>>(name);
//             }, var);
//         }
//     }
//     if(t)
//     {
//         for (size_t i = 0; i < test.size(); i++)
//         {
//             auto &var = test[i];
//             auto &name = nodeNames[i];
//             std::visit([t, name](auto&& arg) {
//                 t->addExposedField(name, VrmlField::typeOf(arg));
//             }, var);
//         }
//     }
// }

MachineNodeBase::MachineNodeBase(VrmlScene *scene)
: VrmlNodeChildTemplate(scene), m_index(machineNodes.size())
, MachineName(registerField<VrmlSFString>("MachineName"))
, VisualizationType(registerField<VrmlSFString>("VisualizationType"))
, ToolHeadNode(registerField<VrmlSFNode>("ToolHeadNode"))
, TableNode(registerField<VrmlSFNode>("TableNode"))
, AxisOrientations(registerField<VrmlMFVec3f>("AxisOrientations"))
, Offsets(registerField<VrmlMFFloat>("Offsets"))
, AxisNames(registerField<VrmlMFString>("AxisNames"))
, ToolNumberName(registerField<VrmlSFString>("ToolNumberName"))
, ToolLengthName(registerField<VrmlSFString>("ToolLengthName"))
, ToolRadiusName(registerField<VrmlSFString>("ToolRadiusName"))
, AxisNodes(registerField<VrmlMFNode>("AxisNodes"))
, OpcUaToVrml(registerField<VrmlSFFloat>("OpcUaToVrml"))
{
    machineNodes.push_back(this);
}

MachineNodeBase::~MachineNodeBase()
{
    machineNodes.erase(machineNodes.begin() + m_index);
}

VrmlNodeType *MachineNodeBase::defineType(VrmlNodeType *t)
{
    assert(t);

    VrmlNodeChildTemplate::defineType(t); // Parent class
    
    t->addExposedField("MachineName", VrmlField::SFSTRING);
    t->addExposedField("VisualizationType", VrmlField::SFSTRING); //None, Currents, Oct
    t->addExposedField("ToolHeadNode", VrmlField::SFNODE);
    t->addExposedField("TableNode", VrmlField::SFNODE);
    t->addExposedField("Offsets", VrmlField::MFFLOAT);
    t->addExposedField("AxisNames", VrmlField::MFSTRING);
    t->addExposedField("ToolNumberName", VrmlField::SFSTRING);
    t->addExposedField("ToolLengthName", VrmlField::SFSTRING);
    t->addExposedField("ToolRadiusName", VrmlField::SFSTRING);
    t->addExposedField("AxisOrientations", VrmlField::MFVEC3F);
    t->addExposedField("AxisNodes", VrmlField::MFNODE);
    t->addExposedField("OpcUaToVrml", VrmlField::SFFLOAT); 

    return t;
}

// array mode

MachineNodeArrayMode::MachineNodeArrayMode(VrmlScene *scene)
: MachineNodeBase(scene)
, OPCUAAxisIndicees(registerField<VrmlMFInt>("OPCUAAxisIndicees"))
, OPCUAArrayName(registerField<VrmlSFString>("OPCUAArrayName"))
{
}

VrmlNodeType *MachineNodeArrayMode::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;
    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("MachineNodeArrayMode", creator);
    }

    MachineNodeBase::defineType(t); // Parent class
    
    t->addExposedField("OPCUAAxisIndicees", VrmlField::MFINT32);
    t->addExposedField("OPCUAArrayName", VrmlField::SFSTRING);

    return t;
}

VrmlNode *MachineNodeArrayMode::creator(VrmlScene *scene)
{
    return new MachineNodeArrayMode(scene);
}

VrmlNodeType *MachineNodeArrayMode::nodeType() const { return defineType(); };

VrmlNode *MachineNodeArrayMode::cloneMe() const 
{
    return new MachineNodeArrayMode(*this);
}

// single mode

MachineNodeSingleMode::MachineNodeSingleMode(VrmlScene *scene)
: MachineNodeBase(scene)
, OPCUANames(registerField<VrmlMFString>("OPCUANames"))
{
}

VrmlNodeType *MachineNodeSingleMode::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;
    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("MachineNodeSingleMode", creator);
    }

    MachineNodeBase::defineType(t); // Parent class
    
    t->addExposedField("OPCUANames", VrmlField::MFSTRING);

    return t;
}

VrmlNode *MachineNodeSingleMode::creator(VrmlScene *scene)
{
    return new MachineNodeSingleMode(scene);
}

VrmlNodeType *MachineNodeSingleMode::nodeType() const { return defineType(); };

VrmlNode *MachineNodeSingleMode::cloneMe() const 
{
    return new MachineNodeSingleMode(*this);
}

// ToolChangerNode

ToolChangerNode::ToolChangerNode(VrmlScene *scene)
: VrmlNodeChildTemplate(scene), m_index(toolChangers.size())
, arm(registerField<VrmlSFString>("arm"))
, changer(registerField<VrmlSFString>("changer"))
, cover(registerField<VrmlSFString>("cover"))
{
    toolChangers.push_back(this);
}

ToolChangerNode::~ToolChangerNode()
{
    toolChangers.erase(toolChangers.begin() + m_index);
}

VrmlNode *ToolChangerNode::creator(VrmlScene *scene)
{
    return new ToolChangerNode(scene);
}

VrmlNodeType *ToolChangerNode::nodeType() const { return defineType(); };

VrmlNode *ToolChangerNode::cloneMe() const 
{
    return new ToolChangerNode(*this);
}

VrmlNodeType *ToolChangerNode::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;
    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("ToolChangerNode", creator);
    }

    MachineNodeBase::defineType(t); // Parent class
    
    t->addExposedField("arm", VrmlField::SFSTRING);
    t->addExposedField("changer", VrmlField::SFSTRING);
    t->addExposedField("cover", VrmlField::SFSTRING);

    return t;
}

// ToolChangerNodeVrml 

ToolChangerNodeVrml::ToolChangerNodeVrml(VrmlScene *scene)
: VrmlNodeChildTemplate(scene), m_index(toolChangers.size())
, arm(registerField<VrmlSFNode>("arm"))
{
    toolChangersVrml.push_back(this);
}

ToolChangerNodeVrml::~ToolChangerNodeVrml()
{
    toolChangersVrml.erase(toolChangersVrml.begin() + m_index);
}

VrmlNode *ToolChangerNodeVrml::creator(VrmlScene *scene)
{
    return new ToolChangerNodeVrml(scene);
}

VrmlNodeType *ToolChangerNodeVrml::nodeType() const { return defineType(); };

VrmlNode *ToolChangerNodeVrml::cloneMe() const 
{
    return new ToolChangerNodeVrml(*this);
}

VrmlNodeType *ToolChangerNodeVrml::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;
    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("ToolChangerNodeVrml", creator);
    }

    MachineNodeBase::defineType(t); // Parent class
    
    t->addExposedField("arm", VrmlField::SFNODE);
    return t;
}

std::vector<ToolChangerNodeVrml *> toolChangersVrml;



// MachineNode dummy

MachineNode::MachineNode(VrmlScene *scene)
: VrmlNodeChildTemplate(scene), m_index(toolChangers.size())
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

    MachineNodeBase::defineType(t); // Parent class
    
    return t;
}