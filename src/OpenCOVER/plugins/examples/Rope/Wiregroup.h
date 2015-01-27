/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Rope_PLUGIN_WIREGROUP_H
#define _Rope_PLUGIN_WIREGROUP_H

#include <cover/coVRPluginSupport.h>
#include <osg/Matrix>
#include <osg/Material>
#include <osg/Billboard>
#include <osg/Camera>
#include <osg/GL>
#include <osg/Group>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/PositionAttitudeTransform>
#include <osg/TexGen>
#include <osg/TexEnv>
#include <osg/TexMat>
#include <osg/TexEnvCombine>
#include <osg/Texture>
#include <osg/TextureCubeMap>
#include <osg/TextureRectangle>
#include <osg/Texture2D>
#include <osg/Geode>
#include <osg/ShapeDrawable>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/CullFace>
#include <osg/BlendFunc>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osg/Fog>
#include <osg/AlphaFunc>
#include <osg/ColorMask>
#include <osgDB/ReadFile>
#include <osg/Program>
#include <osg/Shader>
#include <osg/Point>

#include <cover/coTabletUI.h>
#include <QStringList>
#include <QMap>

#include "rShared.h"
#include "Wire.h"

class Wiregroup : public rShared, public coTUIListener
{
public:
    Wiregroup(rShared *daddy, int id, float lot, coTUIFrame *frame, coTUIComboBox *box);
    ~Wiregroup();
    void createGeom();
    void coverStuff(void);
    int addWire(float posR, float posAngle, float R, coTUIFrame *frame, coTUIComboBox *box);
    xercesc::DOMElement *Save(xercesc::DOMDocument &document);
    void Load(xercesc::DOMElement *rNode);
    void addManipulation(coTUIFrame *frame);
    void delManipulation(void);
    void tabletPressEvent(coTUIElement *tabUI);
    void tabletEvent(coTUIElement *tabUI);
    void setColor(osg::Vec4 color);
    void recurseSetLenFactor(float val);

    static const int maxWires = 32;
    Wire *Wires[maxWires];
    int numWires;

private:
    coTUILabel *LenSliderL;
    coTUILabel *LOTSliderL;
    coTUILabel *colorL;
    coTUILabel *myL1;
    coTUIFloatSlider *myLenSlider;
    coTUIFloatSlider *myLOTSlider;
    coTUILabel *sonL1;
    coTUILabel *sonL2;
    coTUIFloatSlider *sonLenSlider;
    coTUIButton *sonColorButton;
    coTUILabel *sgLenSliderL[maxWires];
    coTUIFloatSlider *sgLenSlider[maxWires];
    coTUIButton *sgColorButton[maxWires];
};
#endif // _Rope_PLUGIN_WIREGROUP_H
