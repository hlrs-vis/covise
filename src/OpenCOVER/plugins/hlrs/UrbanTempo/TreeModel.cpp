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
#include "TreeModel.h"
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

TreeModel::TreeModel()
{
}

TreeModel::TreeModel(std::string configName)
{
    std::vector<double> defaultVec{ 0.0, 0.0, 0.0 };
    auto configSpeciesName = UrbanTempo::instance()->configString(configName, "species_name", "default");
    auto configTreeModel = UrbanTempo::instance()->configString(configName, "model_path", "default");
    auto configSeason = UrbanTempo::instance()->configString(configName, "season", "default");
    auto configLOD = UrbanTempo::instance()->configInt(configName, "LOD", 0);
    auto configRotation = UrbanTempo::instance()->configFloatArray(configName, "rotation", defaultVec);
    auto configTranslation = UrbanTempo::instance()->configFloatArray(configName, "translation", defaultVec);
    osg::Vec3 translation(configTranslation->value()[0], configTranslation->value()[1], configTranslation->value()[2]);

    speciesName = *configSpeciesName;
    modelPath = *configTreeModel;
    model = osgDB::readNodeFile(modelPath);
    season = UrbanTempo::stringToSeason(*configSeason);
    LOD = *configLOD;
    transform = osg::Matrixd::rotate(osg::DegreesToRadians((*configRotation)[0]), osg::Vec3(1.0f, 0.0f, 0.0f),
        osg::DegreesToRadians((*configRotation)[1]), osg::Vec3(0.0f, 1.0f, 0.0f),
        osg::DegreesToRadians((*configRotation)[2]), osg::Vec3(0.0f, 0.0f, 1.0f));
    transform.postMultTranslate(translation);
    if (model)
    {
        osg::ComputeBoundsVisitor cbv;
        model->accept(cbv);
        osg::BoundingBox bb = cbv.getBoundingBox();
        osg::Vec3 size(bb.xMax() - bb.xMin(), bb.yMax() - bb.yMin(), bb.zMax() - bb.zMin());
        if((*configRotation)[0]!=0.0)
            height = size[1]; // hack, if y is up, use y as height, otherwise z
        else
            height = size[2];
    }
}

TreeModel::~TreeModel()
{
}