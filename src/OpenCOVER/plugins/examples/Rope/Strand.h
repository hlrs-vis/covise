/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Rope_PLUGIN_STRAND_H
#define _Rope_PLUGIN_STRAND_H

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
#include "Wiregroup.h"

#define MAX_WIRES 64

class Strand : public rShared, public coTUIListener
{

public:
    Strand(rShared *daddy, int id, float R, float angle, coTUIFrame *frame, coTUIComboBox *box);
    ~Strand();

    void createGeom();
    int addWiregroup(float lot, coTUIFrame *frame, coTUIComboBox *box);
    xercesc::DOMElement *Save(xercesc::DOMDocument &document);
    void Load(xercesc::DOMElement *rNode);
    void addManipulation(coTUIFrame *frame);
    void delManipulation(void);
    void tabletPressEvent(coTUIElement *tabUI);
    void tabletEvent(coTUIElement *tabUI);
    void setColor(osg::Vec4 color);
    void rotate(float lengthOfTwist, bool stateLOT);
    void recurseSetLenFactor(float val);

    static const int maxWiregroups = 32;
    Wiregroup *Wiregroups[maxWiregroups];
    int numWiregroups;

private:
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
    coTUILabel *sgLenSliderL[maxWiregroups];
    coTUIFloatSlider *sgLenSlider[maxWiregroups];
    coTUIFloatSlider *sgLOTSlider[maxWiregroups];
    coTUIButton *sgColorButton[maxWiregroups];
};

#endif // _Rope_PLUGIN_STRAND_H
