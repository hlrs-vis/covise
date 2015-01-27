/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/MatrixTransform>

#include "VariantNode.h"
#include "VRSceneGraph.h"
#include "coVRSelectionManager.h"
#include <cover/coVRPluginSupport.h>
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------

VariantNode::VariantNode(std::string var_Name, osg::Node *node, osg::Node::ParentList pa)
{
    parents = pa;
    varName = var_Name;
    VarNode = new osg::MatrixTransform;
    VarNode->setName(var_Name.c_str());
    VarNode->addChild(node);
    for (osg::Node::ParentList::iterator parent = parents.begin(); parent != parents.end(); ++parent)
    {
        (*parent)->removeChild(node);
    }
    instCounter = 1;
    origin_matrix = VarNode->getMatrix();
    createVRLabel();
}
//------------------------------------------------------------------------------

VariantNode::~VariantNode()
{
    delete VarLabel;
}
//------------------------------------------------------------------------------

void VariantNode::AddToScenegraph()
{
    for (osg::Node::ParentList::iterator parent = parents.begin(); parent != parents.end(); ++parent)
        (*parent)->addChild(VarNode.get());
}
//------------------------------------------------------------------------------

void VariantNode::removeFromScenegraph(osg::Node *node)
{
    for (osg::Node::ParentList::iterator parent = parents.begin(); parent != parents.end(); ++parent)
    {
        (*parent)->removeChild(VarNode.get());
        (*parent)->addChild(node);
    }
}
//------------------------------------------------------------------------------

void VariantNode::attachNode(osg::Node *node)
{
    VarNode->addChild(node);
    instCounter++;
    opencover::cover->getObjectsRoot()->removeChild(node);
}
//------------------------------------------------------------------------------

void VariantNode::dec_Counter()
{
    instCounter--;
}
//------------------------------------------------------------------------------

osg::Node *VariantNode::getNode()
{
    return VarNode.get();
}
//------------------------------------------------------------------------------

osg::Matrix VariantNode::getOriginMatrix()
{
    return origin_matrix;
}
//------------------------------------------------------------------------------

int VariantNode::numNodes()
{
    return instCounter;
}
//------------------------------------------------------------------------------

void VariantNode::createVRLabel()
{
    //Create VRLabel
    osg::MatrixTransform *mtn = new osg::MatrixTransform;
    mtn->setName("Label");
    mtn->setMatrix((mtn->getMatrix()).scale(0.1, 0.1, 0.1));
    VarLabel = new opencover::coVRLabel(varName.c_str(), 5, 10.0, osg::Vec4(1, 1, 1, 1), osg::Vec4(0.1, 0.1, 0.1, 1));
    VarLabel->reAttachTo(mtn);
    VarLabel->setPosition(VarNode->getBound()._center);
    VarNode->addChild(mtn);
}
//------------------------------------------------------------------------------

void VariantNode::showVRLabel()
{
    VarLabel->show();
}

void VariantNode::hideVRLabel()
{
    VarLabel->hide();
}

void VariantNode::printMatrix(osg::MatrixTransform *mt)
{
    osg::Matrix ma = mt->getMatrix();
    cout << "/----------------------- " << endl;
    cout << ma(0, 0) << " " << ma(0, 1) << " " << ma(0, 2) << " " << ma(0, 3) << endl;
    cout << ma(1, 0) << " " << ma(1, 1) << " " << ma(1, 2) << " " << ma(1, 3) << endl;
    cout << ma(2, 0) << " " << ma(2, 1) << " " << ma(2, 2) << " " << ma(2, 3) << endl;
    cout << ma(3, 0) << " " << ma(3, 1) << " " << ma(3, 2) << " " << ma(3, 3) << endl;
    cout << "/-----------------------  " << endl;
}
