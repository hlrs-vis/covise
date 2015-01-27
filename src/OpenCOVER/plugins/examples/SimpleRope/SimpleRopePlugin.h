/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Rope_PLUGIN_H
#define _Rope_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: Rope Plugin (creates Ropes)                              **
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
#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <osg/Matrix>
#include <osg/Material>
#include <osg/Billboard>
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

using namespace opencover;
using namespace vrui;

class Wire
{
public:
    Wire(float radius, float length, float slen, float rlength, int numSegents, int numLengthSegments, int WNum, float angle, osg::Group *group);
    ~Wire();
    void setRadius(float radius);
    float getRadius();
    void setAngle(float angle);
    float getAngle();
    void rotate(float startAngle, float lot, float radius);
    float getLengthOfTwist();
    void setlength(float len);
    float getLength();
    void setstrandlength(float len);
    void setropelength(float len);
    void createGeom();
    void setColor(osg::Vec4 color);
    osg::Vec4 getColor();

private:
    float lengthOfTwist;
    int numSegments;
    int numLengthSegments;
    float length; // in % of Strand-Length
    float strandlength;
    float ropelength;
    float radius;
    float angle;
    float radiusb;

    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Material> globalmtl;
    osg::ref_ptr<osg::DrawArrayLengths> primitives;
    osg::ref_ptr<osg::Vec3Array> vert;
    osg::ref_ptr<osg::Vec3Array> normals;
    osg::ref_ptr<osg::Vec4Array> colors;

    std::vector<int> indices;
    std::vector<osg::Vec3> coord;
    std::vector<osg::Vec3> norm;
};

class Strand
{

public:
    Strand(int numWires, float coreRadius, float hullRadius, float length, float rlength, int numSegents, int numLengthSegments, int SNum, osg::Group *group, float lengthOfTwist, float twistRadius);
    ~Strand();
    void rotate(float startAngle, float lengthOfTwist, float radius);
    void setLengthOfTwist(float lot);
    float getLengthOfTwist();
    float getLengthOfTwistWire();
    float getCoreRadius();
    void setropelength(float length);
    float getropelength();
    void setlength(float length);
    float getLength()
    {
        return length;
    };
    void setlengthfact(float);
    void createGeom();
    void setdifferentlength(float length);
    void setColor(osg::Vec4 color);
    void setwirelength(int ind, float length);
    float getWireLength(int index);
    void setwirecolor(int ind, osg::Vec4 color);
    osg::Vec4 getWireColor(int index);
    void setcorecolor(osg::Vec4 color);
    void setcorelengthfact(float len);

private:
    osg::ref_ptr<osg::Group> strandGroup;
    Wire *core;
    Wire *wires[100];
    int numWires;
    float lengthOfTwist;
    float twistRadius;
    float length; // in % of cable-length
    float ropelength;
};

class SimpleRopePlugin : public coVRPlugin, public coTUIListener
{
public:
    SimpleRopePlugin();
    ~SimpleRopePlugin();

    bool init();

    // this will be called in PreFrame
    void preFrame();
    void calcColors();
    void setdifferentlength(float length);

    Strand *core;
    Strand *strands[100];
    //Wire *wires[100];
    osg::ref_ptr<osg::Group> ropeGroup;

    float length; //absolut length
    int nheight;
    int numSegments;
    int numStrands;
    int numWires;
    float radius;
    float segHeight;
    float twist;
    float twistRadius;
    float toolRadiusSquare;
    int numColors;
    int currentMap;
    float topRadius;
    float squareRope;
    float cubicRope;
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
    coTUITab *Tab1;
    coTUITab *Tab2;
    coTUITab *Tab3;
    coTUIFrame *Frame0;
    coTUIFrame *Frame1;
    coTUIFrame *Frame2;
    coTUIFrame *Frame3;
    coTUIFloatSlider *strandLengthOfTwistSlider;
    coTUIFloatSlider *lengthOfTwistSlider;
    coTUIFloatSlider *lengthSlider;
    coTUIFloatSlider *lengthfactSlider;
    coTUITabFolder *TabFolder;
    coTUIColorTriangle *ColorTriangle;
    coTUIButton *parallelleftbutton;
    coTUIButton *parallelrightbutton;
    coTUIButton *crossedleftbutton;
    coTUIButton *crossedrightbutton;
    coTUIButton *Colorbutton;
    coTUIButton *FiftyPercentButton;
    coTUIButton *StrandAndCoreButton;
    coTUIButton *OnlyOneStrandButton;
    coTUIButton *WireColorButton;
    coTUIButton *OnlyOneWireButton;
    coTUIButton *FileButton;
    coTUIButton *LoadButton;
    coTUIButton *OriginButton;
    coTUIButton *OneWireWithCoreButton;
    coTUIComboBox *StrandComboBox;
    coTUIComboBox *WireComboBox;
    coTUILabel *strandLengthOfTwistLabel;
    coTUILabel *lengthOfTwistLabel;
    coTUILabel *lengthLabel;
    coTUILabel *lengthfactLabel;
    coTUILabel *Label1OfTab1;
    coTUILabel *Label2OfTab1;
    coTUILabel *Label0OfTab3;
    coTUILabel *Label1OfTab3;
    coTUILabel *Label2OfTab3;
    coTUILabel *Label3OfTab3;
    coTUILabel *Label14OfTab3;
    coTUILabel *Label15OfTab3;
    FILE *fileParam;

    coTUILabel *heightLabel;
    coTUILabel *radiusLabel;
    coTUILabel *topRadiusLabel;
    coTUILabel *twistRadiusLabel;
    coTUILabel *numSegmentsLabel;
    coTUILabel *numHLabel;
    coTUILabel *numHTab;
    coTUILabel *twistLabel;
    coTUILabel *mapLabel;
    coTUILabel *squareLabel;
    coTUILabel *cubicLabel;
    coTUILabel *toolLabel;
    coTUILabel *toolRadiusLabel;
    coTUIToggleButton *deformButton;
    coTUIToggleButton *wireframeButton;
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
    osg::ref_ptr<osg::Vec3Array> vert;
    osg::ref_ptr<osg::Vec3Array> normals;
    osg::ref_ptr<osg::Vec4Array> colors;
    osg::ref_ptr<osg::MatrixTransform> cylinderTransform;
    osg::ref_ptr<osg::MatrixTransform> sphereTransform;
    osg::ref_ptr<osg::Cylinder> cylinder;
    osg::ref_ptr<osg::Sphere> sphere;
    osg::ref_ptr<osg::Geode> cylinderGeode;
    osg::ref_ptr<osg::Geode> sphereGeode;
};
#endif
