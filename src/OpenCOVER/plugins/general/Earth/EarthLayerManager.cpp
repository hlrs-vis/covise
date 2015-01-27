/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <EarthLayerManager.h>

#include <osgEarth/Map>
#include <osgEarth/Viewpoint>
#include <osgEarthAnnotation/AnnotationNode>

#include <osg/MatrixTransform>

using namespace osgEarth;
using namespace opencover;

EarthLayerManager::EarthLayerManager(coTUITab *et)
    : earthTab(et)
{
    ImageFrame = new coTUIFrame("ImageLayers", et->getID());
    ImageFrame->setPos(0, 0);
    ElevationFrame = new coTUIFrame("ElevationLayers", et->getID());
    ElevationFrame->setPos(1, 0);
    ModelFrame = new coTUIFrame("ModelLayers", et->getID());
    ModelFrame->setPos(0, 1);
}

void EarthLayerManager::setMap(osgEarth::Map *m)
{
    map = m;
    if (!map.valid())
        return;

    osgEarth::ImageLayerVector layers;
    map->getImageLayers(layers);
    for (osgEarth::ImageLayerVector::const_iterator it = layers.begin(); it != layers.end(); ++it)
    {
        EarthImageLayer *il = new EarthImageLayer(ImageFrame, *it, imageLayers.size());
        imageLayers.push_back(il);
    }
    osgEarth::ElevationLayerVector eLayers;
    map->getElevationLayers(eLayers);
    for (osgEarth::ElevationLayerVector::const_iterator it = eLayers.begin(); it != eLayers.end(); ++it)
    {
        EarthElevationLayer *eLayer = new EarthElevationLayer(ElevationFrame, *it, elevationLayers.size());
        elevationLayers.push_back(eLayer);
    }
    osgEarth::ModelLayerVector mLayers;
    map->getModelLayers(mLayers);
    for (osgEarth::ModelLayerVector::const_iterator it = mLayers.begin(); it != mLayers.end(); ++it)
    {
        EarthModelLayer *mLayer = new EarthModelLayer(ModelFrame, *it, modelLayers.size());
        modelLayers.push_back(mLayer);
    }
}
