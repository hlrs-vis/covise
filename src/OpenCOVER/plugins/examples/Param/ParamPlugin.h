/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Param_PLUGIN_H
#define _Param_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: Param Plugin (does nothing)                              **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Nov-01  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>

using namespace vrui;
using namespace opencover;

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

class ParamPlugin : public coVRPlugin, public coTUIListener
{
public:
    ParamPlugin();
    ~ParamPlugin();
    bool init();

    // this will be called in PreFrame
    void preFrame();
    void createGeom();
    void calcColors();

    int nheight;
    int numSegments;
    float radius;
    float segHeight;
    float twist;
    float twistRadius;
    float toolRadiusSquare;
    int numColors;
    int currentMap;
    float topRadius;
    float squareParam;
    float cubicParam;
    int currentTool;
    enum
    {
        CYLINDER_TOOL = 0,
        SPHERE_TOOL
    };
    QStringList mapNames;
    QMap<QString, int> mapSize;
    QMap<QString, float *> mapValues;

    typedef float FlColor[5];

private:
    void tabletPressEvent(coTUIElement *);
    void tabletEvent(coTUIElement *);
    void deleteColorMap(const QString &);
    void readConfig();
    FlColor *interpolateColormap(FlColor map[], int numSteps);
    osg::Vec4 getColor(float pos);

    coTUITab *paramTab;
    coTUILabel *heightLabel;
    coTUILabel *radiusLabel;
    coTUILabel *topRadiusLabel;
    coTUILabel *twistRadiusLabel;
    coTUILabel *numSegmentsLabel;
    coTUILabel *numHLabel;
    coTUILabel *twistLabel;
    coTUILabel *mapLabel;
    coTUILabel *squareLabel;
    coTUILabel *cubicLabel;
    coTUILabel *toolLabel;
    coTUILabel *toolRadiusLabel;
    coTUIToggleButton *deformButton;
    coTUIToggleButton *wireframeButton;
    coTUIFloatSlider *radiusSlider;
    coTUIFloatSlider *topRadiusSlider;
    coTUIFloatSlider *twistRadiusSlider;
    coTUIFloatSlider *toolRadiusSlider;
    coTUIEditFloatField *heightEdit;
    coTUIEditFloatField *squareEdit;
    coTUIEditFloatField *cubicEdit;
    coTUISlider *numSegmentsSlider;
    coTUISlider *numHSlider;
    coTUIFloatSlider *twistSlider;
    coTUIComboBox *mapChoice;
    coTUIComboBox *toolChoice;

    coTrackerButtonInteraction *interactionA; // push
    coTrackerButtonInteraction *interactionB; // pull

    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Material> globalmtl;
    osg::ref_ptr<osg::DrawArrayLengths> primitives;
    osg::ref_ptr<osg::UShortArray> indices;
    osg::ref_ptr<osg::Vec3Array> vert;
    osg::ref_ptr<osg::UShortArray> nindices;
    osg::ref_ptr<osg::Vec3Array> normals;
    osg::ref_ptr<osg::UShortArray> cindices;
    osg::ref_ptr<osg::Vec4Array> colors;
    osg::ref_ptr<osg::MatrixTransform> cylinderTransform;
    osg::ref_ptr<osg::MatrixTransform> sphereTransform;
    osg::ref_ptr<osg::Cylinder> cylinder;
    osg::ref_ptr<osg::Sphere> sphere;
    osg::ref_ptr<osg::Geode> cylinderGeode;
    osg::ref_ptr<osg::Geode> sphereGeode;
};
#endif
