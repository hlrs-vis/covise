/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "VrmlNodeGeoData.h"

#include <geodata/GeoData.h>

#include "GeoDataLoader.h"

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeGeoData(scene);
}

VrmlNodeGeoData::VrmlNodeGeoData(VrmlScene *scene)
    : VrmlNodeChild(scene, typeName())
    , d_offset(0, 0, 0)
{
}

VrmlNodeGeoData::VrmlNodeGeoData(const VrmlNodeGeoData &n)
    : VrmlNodeChild(n)
    , d_offset(0, 0, 0)
{
}

void VrmlNodeGeoData::initFields(VrmlNodeGeoData *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t); // Parent class
    initFieldsHelper(node, t,
        field("offset", node->d_offset, [node](auto f)
            { opencover::GeoData::instance()->setProjectOffset(osg::Vec3(node->d_offset.get()[0], node->d_offset.get()[1], node->d_offset.get()[2])); }),
        field("offsetName", node->d_offsetName, [node](auto f)
            {
        auto loader = GeoDataLoader::instance();
        auto datasets = loader->getDatasets();
        auto dataset = std::find_if(datasets.begin(), datasets.end(), [node](const GeoDataLoader::DatasetInfo &d)
            { return d.name == node->d_offsetName.get(); });
                if (dataset == datasets.end())
                {
                    std::cerr << "[VrmlNodeGeoData::initFields] GeoData: invalid offsetName '" << node->d_offsetName.get() << "'." << std::endl;
                    return;
                }
                osg::Vec3 origin = osg::Vec3(dataset->easting, dataset->northing, dataset->altitude);
                opencover::GeoData::instance()->setProjectOffset(origin); }),
        field("regions", node->d_regions, [node](auto f)
            {
        auto geoData = GeoDataLoader::instance();
        for (int i = 0; i < node->d_regions.size(); i++)
        {
            auto s = node->d_regions.get(i);
            if (strcmp(s, ALL_REGIONS_STRING) == 0)
            {
                geoData->setAllRegionsEnabled(true);
            }
            geoData->setRegionEnabled(s, true);
        } }),
        field("showTerrain", node->d_showTerrain, [node](auto f)
            { GeoDataLoader::instance()->setShowBuildings(node->d_showTerrain.get()); }),
        field("showLabels", node->d_showLabels, [node](auto f)
            { GeoDataLoader::instance()->setShowLabels(node->d_showLabels.get()); }),
        field("showBuildings", node->d_showBuildings, [node](auto f)
            { GeoDataLoader::instance()->setShowBuildings(node->d_showBuildings.get()); }));
}

const char *VrmlNodeGeoData::typeName()
{
    return "GeoData";
}
