#include "VrmlToolChanger.h"
#include <cassert>

#include <plugins/general/Vrml97/ViewerObject.h>
#include <vrml97/vrml/VrmlNodeGroup.h>

using namespace vrml;

std::set<ToolChangerNode *> toolChangers;


// ToolChangerNode

void ToolChangerNode::initFields(ToolChangerNode *node, VrmlNodeType *t) {
    initFieldsHelper(node, t,
        field("arm", &ToolChangerNode::arm),
        field("changer", &ToolChangerNode::changer),
        field("cover", &ToolChangerNode::cover),
        field("toolHeadNode", &ToolChangerNode::toolHead),
        field("toolMagazineName", &ToolChangerNode::toolMagazineName)
    );
}

ToolChangerNode::ToolChangerNode(VrmlScene *scene)
: VrmlNodeChild(scene, typeName())
{
    toolChangers.emplace(this);
}

ToolChangerNode::~ToolChangerNode()
{
    toolChangers.erase(this);
}

osg::MatrixTransform *toOsg(VrmlNode *node)
{
    auto g = node->as<VrmlNodeGroup>();
    if(!g)
        return nullptr;
    auto vo = g->getViewerObject();
    if(!vo)
        return nullptr;
    auto pNode = ((osgViewerObject *)vo)->pNode;
    if(!pNode)
        return nullptr;
    auto trans = pNode->asTransform();
    if(!trans)
        return nullptr;
    return trans->asMatrixTransform();
}