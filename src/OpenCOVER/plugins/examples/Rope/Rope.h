/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Rope_PLUGIN_ROPE_H
#define _Rope_PLUGIN_ROPE_H

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
#include "strandgroup.h"

class Rope : public rShared, public coTUIListener
{
public:
    int numStrandgroups;
    static const int maxStrandgroups = 16;
    class Strandgroup *Strandgroups[maxStrandgroups];

    Rope(int id, coTUIFrame *frame, coTUIComboBox *box);
    ~Rope(void);

    bool init();

    float setLength(float len);
    float getLength(void);
    void createGeom();
    void Load(xercesc::DOMElement *rNode);
    int addStrandgroup(float lot, coTUIFrame *frame, coTUIComboBox *box);
    xercesc::DOMElement *Save(xercesc::DOMDocument &document);
    void addManipulation(coTUIFrame *frame);
    void delManipulation(void);
    void tabletPressEvent(coTUIElement *tabUI);
    void tabletEvent(coTUIElement *tabUI);

    void setColor(osg::Vec4 color);
    void recurseSetLenFactor(float val);
    void TestRope1(void); // just 4 testing ...
    void AlbertRope(void); // just 4 testing ...

    osg::ref_ptr<osg::Group> ropeGroup;

private:
    FILE *fileParam;

    float length; // Laenge in [mm]

    coTUILabel *ropeL11;
    coTUILabel *ropeL12;
    coTUILabel *ropeL13;
    coTUILabel *ropeL14;
    coTUILabel *ropeL15;
    coTUILabel *ropeL16;
    coTUILabel *ropeL17;
    coTUILabel *ropeL18;
    coTUILabel *ropeL19;
    coTUIEditFloatField *ropePosR;
    coTUIEditFloatField *ropePosA;
    coTUIEditIntField *ropeNumS;
    coTUIEditFloatField *ropeSHeight;
    coTUIEditFloatField *ropeOx;
    coTUIEditFloatField *ropeOy;
    coTUIEditFloatField *ropeOz;
    coTUILabel *LenSliderL;
    coTUILabel *LOTSliderL;
    coTUILabel *colorL;
    coTUILabel *myL1;
    coTUIFloatSlider *myLenSlider;
    coTUILabel *sonL1;
    coTUILabel *sonL2;
    coTUIFloatSlider *sonLenSlider;
    coTUIFloatSlider *sonLOTSlider;
    coTUIButton *sonColorButton;
    coTUILabel *sgLenSliderL[maxStrandgroups];
    coTUIFloatSlider *sgLenSlider[maxStrandgroups];
    coTUIFloatSlider *sgLOTSlider[maxStrandgroups];
    coTUIButton *sgColorButton[maxStrandgroups];
};
#endif // _Rope_PLUGIN_ROPE_H
