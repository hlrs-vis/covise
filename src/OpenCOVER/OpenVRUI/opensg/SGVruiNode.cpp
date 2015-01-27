/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/opensg/SGVruiNode.h>

#include <OpenVRUI/opensg/SGVruiMatrix.h>

#include <OpenSG/OSGMatrix.h>
#include <OpenSG/OSGSimpleAttachments.h>
#include <OpenSG/OSGComponentTransform.h>

OSG_USING_NAMESPACE

SGVruiNode::SGVruiNode(const NodePtr &node)
{
    parent = 0;
    this->node = node;
    addRefCP(node);
}

SGVruiNode::~SGVruiNode()
{
    delete parent;
    subRefCP(node);
}

void SGVruiNode::addChild(vruiNode *node)
{
    SGVruiNode *sgNode = dynamic_cast<SGVruiNode *>(node);
    beginEditCP(this->node, Node::ChildrenFieldMask);
    this->node->addChild(sgNode->getNodePtr());
    endEditCP(this->node, Node::ChildrenFieldMask);
}

void SGVruiNode::insertChild(int location, vruiNode *node)
{
    SGVruiNode *sgNode = dynamic_cast<SGVruiNode *>(node);
    beginEditCP(this->node, Node::ChildrenFieldMask);
    this->node->insertChild(location, sgNode->getNodePtr());
    endEditCP(this->node, Node::ChildrenFieldMask);
}

void SGVruiNode::removeChild(vruiNode *node)
{
    SGVruiNode *sgNode = dynamic_cast<SGVruiNode *>(node);
    beginEditCP(this->node, Node::ChildrenFieldMask);
    this->node->subChild(sgNode->getNodePtr());
    endEditCP(this->node, Node::ChildrenFieldMask);
}

int SGVruiNode::getNumParents() const
{
    return (this->node->getParent() != NullFC ? 1 : 0);
}

vruiNode *SGVruiNode::getParent(int)
{
    if (!parent)
    {
        parent = new SGVruiNode(this->node->getParent());
    }
    else
    {
        NodePtr parentNode = this->node->getParent();
        if (parentNode != osg::NullFC)
        {
            parent->node = parentNode;
        }
        else
        {
            delete parent;
            parent = 0;
        }
    }

    return parent;
}

NodePtr &SGVruiNode::getNodePtr()
{
    return this->node;
}

void SGVruiNode::setName(const std::string &name)
{
    beginEditCP(node);
    osg::setName(node, name);
    endEditCP(node);
}

std::string SGVruiNode::getName() const
{
    return osg::getName(node);
}

void SGVruiNode::removeAllParents()
{

    NodePtr parent = this->node->getParent();
    beginEditCP(parent, Node::ChildrenFieldMask);
    parent->subChild(this->node);
    endEditCP(parent, Node::ChildrenFieldMask);
}

void SGVruiNode::removeAllChildren()
{

    beginEditCP(node, Node::ChildrenFieldMask);
    while (node->getNChildren())
    {
        node->subChild(0);
    }
    endEditCP(node, Node::ChildrenFieldMask);
}

void SGVruiNode::convertToWorld(vruiMatrix *matrix)
{

    SGVruiMatrix *mat = dynamic_cast<SGVruiMatrix *>(matrix);
    Matrix4d returnMatrix = mat->getMatrix();
    SGVruiNode *parent = this;
    while (parent != 0)
    {
        ComponentTransformPtr transform = ComponentTransformPtr::dcast(parent->getNodePtr());
        if (transform != NullFC)
        {
            Matrix4d transformMatrix;
            Real32 *m = transform->getMatrix().getValues();
            transformMatrix.setValue(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
            returnMatrix.mult(transformMatrix);
        }
        parent = dynamic_cast<SGVruiNode *>(parent->getParent());
    }
    mat->setMatrix(returnMatrix);
}

vruiUserData *SGVruiNode::getUserData(const std::string & /*name*/)
{
    //FIXME
    //   Referenced * nodeData = this->node->getUserData();
    //   if (nodeData)
    //   {
    //     OSGVruiUserDataCollection * collection = dynamic_cast<OSGVruiUserDataCollection *>(nodeData);
    //     if (collection) return collection->getUserData(name);
    //   }

    return 0;
}

void SGVruiNode::setUserData(const string & /*name*/, vruiUserData * /*data*/)
{

    //FIXME
    //   Referenced * nodeData = this->node->getUserData();
    //   if (nodeData)
    //   {
    //     OSGVruiUserDataCollection * collection = dynamic_cast<OSGVruiUserDataCollection *>(nodeData);
    //     if (collection)
    //       collection->setUserData(name, data);
    //     else
    //       VRUILOG("OSGVruiNode::setUserData err: unknown node attachment, cannot attach myself")
    //   }
    //   else
    //   {
    //     OSGVruiUserDataCollection * collection = new OSGVruiUserDataCollection();
    //     collection->ref(); // erzeugt ein Memoryleak aber das ist verschmerzbar, ansonsten gibt es Probleme beim Löschen von Knoten
    //     collection->setUserData(name, data);
    //     this->node->setUserData(collection);
    //   }
}
