/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/vsg/VSGVruiNode.h>
#include <OpenVRUI/vsg/VSGVruiMatrix.h>

#include <vsg/nodes/MatrixTransform.h>

#include <OpenVRUI/vsg/VSGVruiUserDataCollection.h>

#include <OpenVRUI/util/vruiLog.h>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

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
    vsg::ref_ptr<vsg::Node> newNode = (dynamic_cast<VSGVruiNode*>(node))->node;
    if (!newNode.get())
        return;
    vruiRendererInterface::the()->compileNode(node);
    vsg::Group* group = dynamic_cast<vsg::Group*>(this->node.get());
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

    if (!vsgNode)
        return;
    vsg::Group* group = dynamic_cast<vsg::Group*>(this->node.get());
    //compile new nodes
    vruiRendererInterface::the()->compileNode(node);
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
    if (nodePath.size() >= 2)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

vruiNode *VSGVruiNode::getParent(int parentNumber)
{
    if (getNumParents())
    {
        vsg::ref_ptr<vsg::Node> parentNode(const_cast<vsg::Node*>(nodePath[nodePath.size() - 2]));
        if (!parent)
        {
            parent = new VSGVruiNode(parentNode);
            parent->setNodePath(std::vector<const vsg::Node*>(nodePath.begin(), nodePath.end() - 1));
            this->node->setValue("nodePath", nodePath);
        }
        else
        {
            if (parentNode.valid())
            {
                parent->node = parentNode;
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
    /*cerr << "undefined vruiNode *VSGVruiNode::getParent(int parentNumber)" << endl;
    return nullptr;*/
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
    /*std::vector<const vsg::Node*> currentNodePath;
    if (this->node->getValue("nodePath", currentNodePath))
    {
        Node* parentNode = const_cast<vsg::Node*>(currentNodePath[currentNodePath.size() - 2]);
        Group* parentGroup = dynamic_cast<Group*>(parentNode);

        for (auto it = parentGroup->children.begin(); it != parentGroup->children.end(); it++)
        {
            if ((*it).get() == this->node.get())
            {
                parentGroup->children.erase(it);
                return;
            }
            else
            {
                cerr << "Child not found in the parent group" << endl;
            }
        }
    }
    else
    {
        cerr << "undefined VSGVruiNode::removeAllParents()" << endl;
    }*/
}

void VSGVruiNode::removeAllChildren()
{
    vsg::Group* group = dynamic_cast<vsg::Group*>(this->node.get());
    group->children.clear();
}

void VSGVruiNode::convertToWorld(vruiMatrix *matrix)
{
    
    cerr << "undefined VSGVruiNode::convertToWorld(vruiMatrix *matrix)" << endl;
    VSGVruiMatrix* mat = dynamic_cast<VSGVruiMatrix*>(matrix);
    dmat4 returnMatrix = mat->getMatrix();

    /*
    VSGVruiNode *parent = this;
    while (parent != 0)
    {
        MatrixTransform *transform = dynamic_cast<MatrixTransform *>(parent->getNodePtr());
        if (transform)
        {
            dmat4 transformMatrix = transform->matrix;
            returnMatrix = transformMatrix * returnMatrix;
            // returnMatrix.postMult(transformMatrix);
        }
        parent = dynamic_cast<VSGVruiNode *>(parent->getParent());
    }
    */
    
    std::vector<const vsg::Node*> currentNodePath;
    if (this->node->getValue("nodePath", currentNodePath))
    { 
        returnMatrix = computeTransform(currentNodePath);
        cerr << "Computing the localToWorld Transform of selected node " << endl;
    }
    mat->setMatrix(returnMatrix);
    
}

vruiUserData *VSGVruiNode::getUserData(const std::string &name)
{
    return VSGVruiUserDataCollection::getUserData(this->node, name);
}

void VSGVruiNode::setUserData(const string &name, vruiUserData *data)
{
    VSGVruiUserDataCollection::setUserData(this->node, name, data);
}

void VSGVruiNode::setNodePath(std::vector<const vsg::Node*> hitNodePath)
{
    nodePath = hitNodePath;
}
}
