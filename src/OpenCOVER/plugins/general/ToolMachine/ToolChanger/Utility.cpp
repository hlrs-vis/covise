#include "Utility.h"
#include <osg/MatrixTransform>

AnimationManagerFinder::AnimationManagerFinder(): osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN) {}
void AnimationManagerFinder::apply(osg::Node& node) {

    if (m_am.valid())
        return;

    if (node.getUpdateCallback()) {       
        m_am = dynamic_cast<osgAnimation::BasicAnimationManager*>(node.getUpdateCallback());
        return;
    }

    traverse(node);
}


TransformFinder::TransformFinder(const std::string &name)
: osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN)
, m_name(name) {}

void TransformFinder::apply(osg::Node& node) {

    if (trans)
        return;

    if (node.asTransform() && node.asTransform()->asMatrixTransform() && node.getName() == m_name) {       
        trans = node.asTransform()->asMatrixTransform();
        return;
    }
    traverse(node);
}

osg::MatrixTransform *findTransform(osg::Node *node, const std::string &name)
{
    TransformFinder f(name);
    node->accept(f);
    return f.trans;
}