/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/osg/OSGVruiNode.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <osg/MatrixTransform>

#include <OpenVRUI/osg/OSGVruiUserDataCollection.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace osg;
using namespace std;

namespace vrui
{

OSGVruiNode::OSGVruiNode(Node *node)
{
    parent = 0;
    this->node = node;
    node->ref();
}

OSGVruiNode::~OSGVruiNode()
{
    node->unref();
    delete parent;
}

void OSGVruiNode::addChild(vruiNode *node)
{
    OSGVruiNode *osgNode = dynamic_cast<OSGVruiNode *>(node);
    if (!this->node->asGroup()->containsNode(osgNode->getNodePtr()))
        this->node->asGroup()->addChild(osgNode->getNodePtr());
}

void OSGVruiNode::insertChild(int location, vruiNode *node)
{
    OSGVruiNode *osgNode = dynamic_cast<OSGVruiNode *>(node);
    this->node->asGroup()->insertChild(location, osgNode->getNodePtr());
}

void OSGVruiNode::removeChild(vruiNode *node)
{
    OSGVruiNode *osgNode = dynamic_cast<OSGVruiNode *>(node);
    this->node->asGroup()->removeChild(osgNode->getNodePtr());
}

int OSGVruiNode::getNumParents() const
{
    return this->node->getNumParents();
}

vruiNode *OSGVruiNode::getParent(int parentNumber)
{
    if (this->node->getNumParents())
    {
        if (!parent)
        {
            parent = new OSGVruiNode(this->node->getParent(parentNumber));
        }
        else
        {
            Node *parentNode = this->node->getParent(parentNumber);
            if (parentNode != 0)
            {
                parent->node = parentNode;
                parent->node->ref();
            }
            else
            {
                delete parent;
                parent = 0;
            }
        }
    }
    else
    {
        delete parent;
        parent = 0;
    }

    return parent;
}

Node *OSGVruiNode::getNodePtr()
{
    return this->node.get();
}

void OSGVruiNode::setName(const string &name)
{
    this->node->setName(name);
}

std::string OSGVruiNode::getName() const
{
    return this->node->getName();
}

void OSGVruiNode::removeAllParents()
{

    Node::ParentList parents = this->node->getParents();

    for (Node::ParentList::iterator i = parents.begin(); i != parents.end(); ++i)
    {
        (*i)->removeChild(this->node.get());
    }
}

void OSGVruiNode::removeAllChildren()
{

    Group *group = this->node->asGroup();

    if (group || (group->getNumChildren() > 0))
    {
        group->removeChild(0, group->getNumChildren());
    }
}

void OSGVruiNode::convertToWorld(vruiMatrix *matrix)
{

    OSGVruiMatrix *mat = dynamic_cast<OSGVruiMatrix *>(matrix);
    Matrix returnMatrix = mat->getMatrix();
    OSGVruiNode *parent = this;
    while (parent != 0)
    {
        MatrixTransform *transform = dynamic_cast<MatrixTransform *>(parent->getNodePtr());
        if (transform)
        {
            Matrix transformMatrix = transform->getMatrix();
            returnMatrix.postMult(transformMatrix);
        }
        parent = dynamic_cast<OSGVruiNode *>(parent->getParent());
    }
    mat->setMatrix(returnMatrix);
}

vruiUserData *OSGVruiNode::getUserData(const std::string &name)
{

    Referenced *nodeData = this->node->getUserData();
    if (nodeData)
    {
        OSGVruiUserDataCollection *collection = dynamic_cast<OSGVruiUserDataCollection *>(nodeData);
        if (collection)
            return collection->getUserData(name);
    }

    return 0;
}

void OSGVruiNode::setUserData(const string &name, vruiUserData *data)
{

    Referenced *nodeData = this->node->getUserData();
    if (nodeData)
    {
        OSGVruiUserDataCollection *collection = dynamic_cast<OSGVruiUserDataCollection *>(nodeData);
        if (collection)
            collection->setUserData(name, data);
        else
            VRUILOG("OSGVruiNode::setUserData err: unknown node attachment, cannot attach myself")
    }
    else
    {
        OSGVruiUserDataCollection *collection = new OSGVruiUserDataCollection();
        collection->ref(); // erzeugt ein Memoryleak aber das ist verschmerzbar, ansonsten gibt es Probleme beim Löschen von Knoten
        collection->setUserData(name, data);
        this->node->setUserData(collection);
    }
}
}
