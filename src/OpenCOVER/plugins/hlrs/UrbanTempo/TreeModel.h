/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TREEMODEL_H
#define _TREEMODEL_H

#include <string>
#include <cover/coVRPlugin.h>
#include <cover/coVRFileManager.h>
#include <rapidjson/document.h>
#include <osg/Node>
#include <osg/ref_ptr>
#include <gdal_priv.h>

enum Season { 
    Winter,
    Spring,
    Summer,
    Fall
};


class TreeModel
{
public:
    TreeModel();
    TreeModel(std::string configName);
    ~TreeModel();
    std::string speciesName;
    std::string modelPath;
    Season season;
    int LOD;
    osg::ref_ptr<osg::Node> model;
    osg::Matrix transform;
    float height;
};

#endif

