/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Rope_PLUGIN_WIRE_H
#define _Rope_PLUGIN_WIRE_H

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
using namespace opencover;

class Wire : public rShared
{
public:
    Wire(rShared *daddy, int id, float posR, float posAngle, float R, coTUIFrame *frame, coTUIComboBox *box);
    ~Wire();
    void rotate(float posR, float posAngle, float lengthOfTwist, bool stateLOT, osg::Vec3 orient);

    void setColor(osg::Vec4 color);
    osg::Vec4 getColor(void);
    void coverStuff(void);
    void createGeom(void);
    void cutWire(osg::Vec3 normal, float dist);
    void setWireRadius(float R);
    xercesc::DOMElement *Save(xercesc::DOMDocument &document);
    void Load(xercesc::DOMElement *Node);

private:
    // rei, TODO, keine Ahnung ... int numLengthSegments;
    // der radius und die numSegments werden spaeter durch eine
    // Punktewolke ersetzt, welche die Grundflaeche bildet
    float R; // Radius des Drahtes -->

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
#endif // _Rope_PLUGIN_WIRE_H
