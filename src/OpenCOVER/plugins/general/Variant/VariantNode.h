/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* 
 * File:   VariantNode.h
 * Author: Gottlieb A.
 *
 * Created on 14. August 2009, 09:45
 */

#ifndef _VARIANTNODE_H
#define _VARIANTNODE_H

#include <osg/Group>
#include <iostream>
#include <cover/coVRPlugin.h>
#include "coVRLabel.h"

using namespace std;

class VariantNode
{
public:
    VariantNode(std::string var_Name, osg::Node *node, osg::Node::ParentList parents);
    ~VariantNode();
    //adding the Group-Node to the Scenegraph:
    void AddToScenegraph();
    void removeFromScenegraph(osg::Node *node);
    void attachNode(osg::Node *node);
    void dec_Counter();
    osg::Node *getNode();
    void printMatrix(osg::MatrixTransform *mt);

    std::string getVarname()
    {
        return varName;
    };
    int numNodes();
    osg::Matrix getOriginMatrix();
    void createVRLabel();
    void showVRLabel();
    void hideVRLabel();

private:
    osg::ref_ptr<osg::MatrixTransform> VarNode;
    std::string varName;
    int instCounter;
    osg::Node::ParentList parents;
    osg::Matrix origin_matrix;
    opencover::coVRLabel *VarLabel;
};

#endif /* _VARIANTNODE_H */
