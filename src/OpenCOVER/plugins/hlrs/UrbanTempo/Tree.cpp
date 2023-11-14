/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                            (C)2023 HLRS  **
 **                                                                          **
 ** Description: Urban Tempo OpenCOVER Plugin                                **
 **                                                                          **
 **                                                                          **
 ** Author: Kilian Türk                                                      **
 **                                                                          **
 ** History:  								                                 **
 ** Feb 2023  v1	    				       		                         **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "UrbanTempo.h"
#include "Tree.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
// #include <OpenVRUI/coMenu.h>
#include <cover/coVRTui.h>
// #include <coVRTui>
#include <curl/curl.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/prettywriter.h> 
#include <rapidjson/rapidjson.h>
#include <iostream>
#include <osgDB/ReadFile>
#include <osg/MatrixTransform>
#include <osg/ref_ptr>
#include <osg/Referenced>
#include <fstream>
#include <sstream>
#include <osg/ComputeBoundsVisitor>
#include <osg/PositionAttitudeTransform>
#include <osg/Vec3d>
#include <unordered_map>

#include <osg/io_utils>

using namespace opencover;

Tree::Tree(std::string configName)
{
    auto configSpeciesName = UrbanTempo::instance()->configString(configName, "species_name", "default");
    speciesName = *configSpeciesName;
    auto configTreeX = UrbanTempo::instance()->configFloat(configName, "x", 0);
    xPos = *configTreeX;
    auto configTreeY = UrbanTempo::instance()->configFloat(configName, "y", 0);
    yPos = *configTreeY;

    auto configHeight = UrbanTempo::instance()->configFloat(configName, "height", 0.0);
    height = *configHeight;
    if (height == 0.0)
        height = UrbanTempo::instance()->defaultHeight;
    
    altitude = 0.0;
    scale = 0.0;
    treeModel = nullptr;
}

// constructor for tree initializing all memeber variables
Tree::Tree(std::string speciesName, double xPos, double yPos, double height, float altitude, double scale, std::shared_ptr<TreeModel> treeModel, std::vector<std::shared_ptr<TreeModel>> treeModelLODs, osg::ref_ptr<osg::MatrixTransform> sceneGraphNode)
{
    this->speciesName = speciesName;
    this->xPos = xPos;
    this->yPos = yPos;
    this->height = height;
    this->altitude = altitude;
    this->scale = scale;
    this->treeModel = treeModel;
    this->treeModelLODs = treeModelLODs;
    this->sceneGraphNode = sceneGraphNode;
}

Tree::Tree(std::string speciesName, double xPos, double yPos, double height)
{
    this->speciesName = speciesName;
    this->xPos = xPos;
    this->yPos = yPos;
    this->height = height;
    this->altitude = 0.0;
    this->scale = 1.0;
    this->treeModel = nullptr;
    this->treeModelLODs = std::vector<std::shared_ptr<TreeModel>>();
    this->sceneGraphNode = nullptr;
}

Tree::~Tree()
{
}

void Tree::updateScale()
{
    scale = this->height / treeModel->height;
}

void Tree::setAltitude(const float altitude)
{
    this->altitude = altitude;
}

double Tree::getDistanceToViewer()
{
    auto invBaseMat = cover->getInvBaseMat();
    auto viewerMat = cover->getViewerMat();
    auto mat1 = viewerMat * invBaseMat;
    auto posViewer = mat1.getTrans();
    auto posTree = sceneGraphNode->getMatrix().getTrans();

    // auto diff = vec2 - vec1;
    // sqrt(diff * diff);
    // distance = sqrt((posViewer));

    return UrbanTempo::getDistance(posViewer, posTree);
}

void Tree::setTreeModelLODs()
{
    // std::vector<std::shared_ptr<TreeModel>> treeModelLODs;
    for (int i = 0; i < UrbanTempo::instance()->treeModels.size(); ++i)
    {
        if (this->speciesName == UrbanTempo::instance()->treeModels[i]->speciesName)
        {
            treeModelLODs.emplace_back(UrbanTempo::instance()->treeModels[i]);
        }
    }

    std::sort(treeModelLODs.begin(), treeModelLODs.end(), [](std::shared_ptr<TreeModel> a, std::shared_ptr<TreeModel> b)
    {
        return a->LOD > b->LOD;
    });
}

void Tree::updateTreeModel()
{
    if (treeModelLODs.size() > 1)
    {
        // double lodMaxDistance = 100;
        // coVRConfig::instance()->setLODScale(100);
        auto lodMaxDistance = coVRConfig::instance()->getLODScale();
        double lodInterval = lodMaxDistance / (treeModelLODs.size() - 1);
        std::vector<double> distanceIntervals;
        for (int i = 1; i < treeModelLODs.size(); ++i)
        {
            distanceIntervals.emplace_back(i * lodInterval);
        }

        std::shared_ptr<TreeModel> newTreeModel = nullptr;
        std::cout << getDistanceToViewer() << "\n";
        auto intervalIterator = std::upper_bound(distanceIntervals.begin(), distanceIntervals.end(), getDistanceToViewer());
        int index = intervalIterator - distanceIntervals.begin();
        newTreeModel = treeModelLODs[index];

        this->treeModel = newTreeModel;
        this->updateScale();
        this->setTransform();
        this->updateSceneGraphNode();
    }
}

void Tree::setTransform()
{
    transform = osg::Matrixd::scale(osg::Vec3d(scale, scale, scale))
        * treeModel->transform
        * osg::Matrix::translate(osg::Vec3d(xPos, yPos, altitude));
}

void Tree::attachToNode(osg::Node* node)
{
    // osg::MatrixTransform* nodeTransform = dynamic_cast<osg::MatrixTransform*>(node);
    osg::MatrixTransform* newNode;
    newNode->setMatrix(transform);
    newNode->addChild(treeModel->model);
    node = newNode;
}

void Tree::updateSceneGraphNode()
{
    this->sceneGraphNode->setMatrix(transform);
    this->sceneGraphNode->removeChild(0, 1);
    this->sceneGraphNode->addChild(treeModel->model);
}