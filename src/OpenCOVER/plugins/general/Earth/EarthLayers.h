/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EarthLayers_H
#define EarthLayers_H 1

#include <cover/coTabletUI.h>

#include <osgEarth/Map>
#include <osgEarth/Layer>

using namespace osgEarth;
using namespace opencover;

//---------------------------------------------------------------------------
class EarthLayer : public coTUIListener
{

public:
    EarthLayer(coTUIFrame *parent, Layer *layer, int LayerNumber);

protected:
    int layerNumber;
    osg::ref_ptr<Layer> layerBase;
    coTUIFrame *parentFrame;
    coTUILabel *layerName;
};

//---------------------------------------------------------------------------
class EarthElevationLayer : public EarthLayer
{

public:
    EarthElevationLayer(coTUIFrame *parent, ElevationLayer *layer, int LayerNumber);

    virtual void tabletEvent(coTUIElement *);

protected:
    osg::ref_ptr<ElevationLayer> layer;
    coTUIToggleButton *visibleCheckBox;
};

class EarthImageLayer : public EarthLayer
{

public:
    EarthImageLayer(coTUIFrame *parent, ImageLayer *layer, int LayerNumber);

    void tabletEvent(coTUIElement *);

protected:
    osg::ref_ptr<ImageLayer> layer;
    coTUIToggleButton *visibleCheckBox;
    coTUIFloatSlider *opacitySlider;
};

class EarthModelLayer : public EarthLayer
{

public:
    EarthModelLayer(coTUIFrame *parent, ModelLayer *layer, int LayerNumber);

    void tabletEvent(coTUIElement *);

protected:
    osg::ref_ptr<ModelLayer> layer;
    coTUIToggleButton *visibleCheckBox;
    coTUIToggleButton *overlayCheckBox;
};

#endif // EarthLayers_H
