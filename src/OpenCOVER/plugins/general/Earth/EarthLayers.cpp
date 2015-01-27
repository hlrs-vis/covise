/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include <EarthLayers.h>

#include <osgEarth/Map>
#include <osgEarth/Viewpoint>
#include <osgEarthAnnotation/AnnotationNode>

#include <osg/MatrixTransform>

using namespace osgEarth;

EarthLayer::EarthLayer(coTUIFrame *parent, Layer *layer, int LayerNumber)
{
    parentFrame = parent;
    layerBase = layer;
    layerNumber = LayerNumber;
    layerName = new coTUILabel("LayerName", parentFrame->getID());
    layerName->setPos(0, LayerNumber);
}

EarthElevationLayer::EarthElevationLayer(coTUIFrame *parent, ElevationLayer *l, int LayerNumber)
    : EarthLayer(parent, l, LayerNumber)
{
    layer = l;
    layerName->setLabel(l->getName());
    visibleCheckBox = new coTUIToggleButton("CheckBox", parentFrame->getID(), layer->getVisible());
    visibleCheckBox->setPos(1, LayerNumber);
    visibleCheckBox->setEventListener(this);
}

void EarthElevationLayer::tabletEvent(coTUIElement *e)
{
    if (e == visibleCheckBox)
    {
        layer->setVisible(visibleCheckBox->getState());
    }
}

EarthImageLayer::EarthImageLayer(coTUIFrame *parent, ImageLayer *l, int LayerNumber)
    : EarthLayer(parent, l, LayerNumber)
{
    layer = l;
    layerName->setLabel(l->getName());
    visibleCheckBox = new coTUIToggleButton("CheckBox", parentFrame->getID(), layer->getVisible());
    visibleCheckBox->setPos(1, LayerNumber);
    visibleCheckBox->setEventListener(this);
    opacitySlider = new coTUIFloatSlider("opacity", parentFrame->getID());
    opacitySlider->setMin(0.0);
    opacitySlider->setMax(1.0);
    opacitySlider->setValue(1.0);
    opacitySlider->setPos(2, LayerNumber);
    opacitySlider->setEventListener(this);
}

void EarthImageLayer::tabletEvent(coTUIElement *e)
{
    if (e == visibleCheckBox)
    {
        layer->setVisible(visibleCheckBox->getState());
    }
    if (e == opacitySlider)
    {
        layer->setOpacity(opacitySlider->getValue());
    }
}

EarthModelLayer::EarthModelLayer(coTUIFrame *parent, ModelLayer *l, int LayerNumber)
    : EarthLayer(parent, l, LayerNumber)
{
    layer = l;
    layerName->setLabel(l->getName());
    visibleCheckBox = new coTUIToggleButton("CheckBox", parentFrame->getID(), layer->getVisible());
    visibleCheckBox->setPos(1, LayerNumber);
    visibleCheckBox->setEventListener(this);

    overlayCheckBox = new coTUIToggleButton("overlay", parentFrame->getID(), layer->getVisible());
    overlayCheckBox->setPos(2, LayerNumber);
    overlayCheckBox->setEventListener(this);
}

void EarthModelLayer::tabletEvent(coTUIElement *e)
{
    if (e == visibleCheckBox)
    {
        layer->setVisible(visibleCheckBox->getState());
    }
    if (e == overlayCheckBox)
    {
        //layer->setOverlay(overlayCheckBox->getState());
        std::cerr << "Earth plugin: overlay: not supported by current osgEarth" << std::endl;
    }
}
