/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiNode.h>
#include <OpenVRUI/vsg/VSGVruiMatrix.h>

#include <vsg/nodes/MatrixTransform.h>

#include <OpenVRUI/vsg/VSGVruiUserDataCollection.h>

#include <OpenVRUI/util/vruiLog.h>

using namespace vsg;
using namespace std;

namespace vrui
{

VSGVruiNode::VSGVruiNode(vsg::ref_ptr<vsg::Node> in_node)
{
    parent = 0;
    node = in_node;
}

VSGVruiNode::~VSGVruiNode()
{
    delete parent;
}

void VSGVruiNode::addChild(vruiNode *node)
{
    vsg::ref_ptr<vsg::Node> newNode = (dynamic_cast<VSGVruiNode *>(node))->node;
    vsg::Group* group = dynamic_cast<vsg::Group *>(this->node.get());
    bool found = false;
    for(auto it = group->children.begin();it != group->children.end(); it++)
    {
        if((*it).get() == newNode.get())
        { 
            found = true;
            break;
        }
    }
    if(!found)
        group->addChild(newNode);
}

void VSGVruiNode::insertChild(int location, vruiNode *node)
{
    VSGVruiNode *vsgNode = dynamic_cast<VSGVruiNode *>(node);
    vsg::Group* group = dynamic_cast<vsg::Group*>(this->node.get());
    group->children.insert(group->children.begin()+location, vsgNode->node);
}

void VSGVruiNode::removeChild(vruiNode *node)
{
    VSGVruiNode *vsgNode = dynamic_cast<VSGVruiNode *>(node);
    vsg::Group* group = dynamic_cast<vsg::Group*>(this->node.get());

    for (auto it = group->children.begin(); it != group->children.end(); it++)
    {
        if ((*it).get() == vsgNode->getNodePtr())
        {
            group->children.erase(it);
            break;
        }
    }
}

int VSGVruiNode::getNumParents() const
{
    cerr << "undefined VSGVruiNode::getNumParents()" << endl;
    return 0;
}

vruiNode *VSGVruiNode::getParent(int parentNumber)
{
    
    cerr << "undefined vruiNode *VSGVruiNode::getParent(int parentNumber)" << endl;
    return nullptr;
}

Node *VSGVruiNode::getNodePtr()
{
    return this->node.get();
}

void VSGVruiNode::setName(const string &name)
{
    this->node->setValue("name",name);
}

std::string VSGVruiNode::getName() const
{
    std::string val;
    this->node->getValue("name",val);
    return val;
}

void VSGVruiNode::removeAllParents()
{

    cerr << "undefined VSGVruiNode::removeAllParents()" << endl;
}

void VSGVruiNode::removeAllChildren()
{
    vsg::Group* group = dynamic_cast<vsg::Group*>(this->node.get());
    group->children.clear();
}

void VSGVruiNode::convertToWorld(vruiMatrix *matrix)
{

    cerr << "undefined VSGVruiNode::convertToWorld(vruiMatrix *matrix)" << endl;
   /* VSGVruiMatrix* mat = dynamic_cast<VSGVruiMatrix*>(matrix);
    Matrix returnMatrix = mat->getMatrix();
    VSGVruiNode *parent = this;
    while (parent != 0)
    {
        MatrixTransform *transform = dynamic_cast<MatrixTransform *>(parent->getNodePtr());
        if (transform)
        {
            Matrix transformMatrix = transform->getMatrix();
            returnMatrix.postMult(transformMatrix);
        }
        parent = dynamic_cast<VSGVruiNode *>(parent->getParent());
    }
    mat->setMatrix(returnMatrix);*/
}

vruiUserData *VSGVruiNode::getUserData(const std::string &name)
{
    return VSGVruiUserDataCollection::getUserData(this->node, name);
}

void VSGVruiNode::setUserData(const string &name, vruiUserData *data)
{
    VSGVruiUserDataCollection::setUserData(this->node, name, data);
}
}
