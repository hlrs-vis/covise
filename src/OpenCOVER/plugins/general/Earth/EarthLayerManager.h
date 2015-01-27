/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EarthLayerManager_H
#define EarthLayerManager_H 1

#include <cover/coTabletUI.h>

#include <osgEarth/Map>
#include <osgEarth/Layer>
#include "EarthLayers.h"
#include <list>

using namespace osgEarth;
using namespace opencover;

//---------------------------------------------------------------------------
class EarthLayerManager : public coTUIListener
{

    enum LayerType
    {
        ELEVATION_LAYERS,
        IMAGE_LAYERS,
        MODEL_LAYERS
    };

public:
    EarthLayerManager(coTUITab *et);
    void setMap(osgEarth::Map *m);

protected:
    coTUITab *earthTab;
    osg::ref_ptr<osgEarth::Map> map;

    typedef std::list<EarthLayer *> LayerList;
    LayerList elevationLayers;
    LayerList imageLayers;
    LayerList modelLayers;

    coTUIFrame *ElevationFrame;
    coTUIFrame *ImageFrame;
    coTUIFrame *ModelFrame;
};

#endif // EarthLayerManager_H
