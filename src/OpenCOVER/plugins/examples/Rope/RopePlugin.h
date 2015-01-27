/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Rope_PLUGIN_H
#define _Rope_PLUGIN_H

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
#include "Rope.h"
#include "Strand.h"
#include "Wire.h"

#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/validators/common/GrammarResolver.hpp>
#include <xercesc/validators/schema/SchemaGrammar.hpp>
using namespace opencover;

class RopePlugin : public coVRPlugin, public coTUIListener
{
public:
    RopePlugin();
    ~RopePlugin();

    bool init();

    // this will be called in PreFrame
    void preFrame();
    void setdifferentlength(float length);
    bool Save(const char *fn);

    osg::ref_ptr<osg::Group> ropeGroup;

    float length; //absolut length
    int nheight;
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
    //QStringList            mapNames;
    //QMap<QString, int>     mapSize;
    //QMap<QString, float*>  mapValues;

    //typedef float FlColor[5];

    int addRope(coTUIFrame *frame, coTUIComboBox *box);
    bool Load(const char *fn);
    void adaptManipulation(coTUIFrame *frame);
    char *nextID(char *p);
    void recurseSetLenFactor(float val);

private:
    static const int maxRopes = 16;
    Rope *Ropes[maxRopes];
    int numRopes;
    void tabletPressEvent(coTUIElement *);
    void tabletEvent(coTUIElement *);
    osg::Vec4 getColor(float pos);
    int rID, sgID, sID, wgID;

    coTUIFileBrowserButton *SaveButton;
    coTUIFileBrowserButton *LoadButton;
    coTUIButton *TestRopeButton;
    coTUIButton *AlbertRopeButton;
    coTUIButton *SaveAlbertRope;
    coTUIButton *LoadAlbertRope;
    coTUIButton *SaveTestRope;
    coTUIButton *LoadTestRope;
    coTUITab *paramTab;
    coTUITabFolder *TabFolder;
    coTUITab *Tab1;
    coTUITab *Tab2;
    coTUITab *Tab3;
    coTUIFrame *Frame0;
    coTUIFrame *Frame1;
    coTUIFrame *Frame2;
    coTUIFrame *Frame3;
    coTUIComboBox *SelComboBox;
    coTUIFileBrowserButton *loadFileBr;
    coTUIColorTriangle *ColorTriangle;

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
