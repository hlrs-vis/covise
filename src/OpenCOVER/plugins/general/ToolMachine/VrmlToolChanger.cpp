#include "VrmlToolChanger.h"
#include <cassert>

#include <plugins/general/Vrml97/ViewerObject.h>
#include <vrml97/vrml/VrmlNodeGroup.h>

using namespace vrml;

std::set<ToolChangerNode *> toolChangers;


// ToolChangerNode

void ToolChangerNode::initFields(ToolChangerNode *node, VrmlNodeType *t) {
    initFieldsHelper(node, t,
        field("arm", node->arm),
        field("changer", node->changer),
        field("cover", node->cover),
        field("toolHeadNode", node->toolHead),
        field("toolMagazineName", node->toolMagazineName)
    );
}

ToolChangerNode::ToolChangerNode(VrmlScene *scene)
: VrmlNodeTemplate(scene, name())
{
    // initFields(this, nullptr);
    toolChangers.emplace(this);
}

ToolChangerNode::~ToolChangerNode()
{
    toolChangers.erase(this);
}

osg::MatrixTransform *toOsg(VrmlNode *node)
{
    auto g = node->toGroup();
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