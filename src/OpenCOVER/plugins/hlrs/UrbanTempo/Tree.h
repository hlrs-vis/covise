/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TREE_H
#define _TREE_H

#include "TreeModel.h"
// #include "TreeBuilder.h"
#include <string>
#include <cover/coVRPlugin.h>
#include <cover/coVRFileManager.h>
#include <rapidjson/document.h>
#include <osg/Node>
#include <osg/ref_ptr>
#include <gdal_priv.h>

// enum Season { 
//     Winter,
//     Spring,
//     Summer,
//     Fall
// };

class Tree
{
private:
  // friend class TreeBuilder;

public:
    Tree(std::string);
    Tree(std::string, double, double, double);
    Tree(std::string, double, double, double, float, double, std::shared_ptr<TreeModel>, std::vector<std::shared_ptr<TreeModel>>, osg::ref_ptr<osg::MatrixTransform> );
    ~Tree();
    std::string speciesName;
    double xPos;
    double yPos;
    float altitude;
    double scale;
    double height;
    osg::Matrix transform;
    std::shared_ptr<TreeModel> treeModel;
    std::vector<std::shared_ptr<TreeModel>> treeModelLODs;
    osg::ref_ptr<osg::MatrixTransform> sceneGraphNode;
    
    void setAltitude(const float);
    void updateScale();
    void updateTreeModel();
    void setTransform();
    double getDistanceToViewer();
    void updateSceneGraphNode();
    void setTreeModelLODs();
};

#endif

