/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Rope_PLUGIN_RSHARED_H
#define _Rope_PLUGIN_RSHARED_H

#include <cover/coVRPluginSupport.h>
#include <cover/coVRTui.h>
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
#include <OpenVRUI/osg/mathUtils.h>
#include <xercesc/dom/DOM.hpp>
#if XERCES_VERSION_MAJOR < 3
#include <xercesc/dom/DOMWriter.hpp>
#else
#include <xercesc/dom/DOMLSSerializer.hpp>
#endif
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/validators/common/GrammarResolver.hpp>
#include <xercesc/validators/schema/SchemaGrammar.hpp>

class rShared
{
public:
    rShared();
    ~rShared();
    void initShared(rShared *daddy, char *name, int id, opencover::coTUIFrame *frame, opencover::coTUIComboBox *box);

    float getLength();
    float setLenFactor(float val);
    float getLenFactor();
    float setLengthOfTwist(float len);
    float getLengthOfTwist(void);
    void setStateLengthOfTwist_On(void);
    void setStateLengthOfTwist_Off(void);
    bool getStateLengthOfTwist(void);
    float setPosAngle(float angle);
    float getPosAngle(void);
    float setPosRadius(float R);
    float getPosRadius(void);
    float getRopeLength(void);
    void setRopeLength(float val);
    void setNumSegments(int val);
    int getNumSegments(void);
    void setSegHeight(float val);
    float getSegHeight(void);
    void setOrientation(osg::Vec3 O);
    osg::Vec3 getOrientation(void);
    float getElemLength(void);
    bool isCore(void);
    void setCore(bool state);
    int getID(void);
    char *getName(void);
    char *getIDName(void);
    char *getIDNamePath(void);
    char *identStr(void);
    int getElemID(void);
    osg::Vec4 getColTr(void);
    opencover::coTUIColorTriangle *colTr;
    float rad2grad(float rad);
    float grad2rad(float grad);

    rShared *daddy;
    osg::Vec3 orientation;
    opencover::coTUIFrame *coverFrame;
    opencover::coTUIComboBox *coverBox;
    osg::ref_ptr<osg::Group> coverGroup;

private:
    float elemLenFactor; // for Strand, Wires
    float elemR; // Pos: Radius from 0/0/0
    float elemAngle; // Pos: angle
    // elemLengthOfTwist = Wert der Schlaglaenge
    // elemFlagLengthOfTwist: Schlaglaenge an oder aus
    // Idee: bei Brueckenseilen gibt es auch parallel-Buendel
    // diese koennten wir damit auch darstellen
    float elemLengthOfTwist; // Schlaglaenge
    bool elemStateLengthOfTwist; // Schlaglaenge an oder aus
    char *elemName;
    char *elemIDName;
    char *elemIDNamePath;
    int elemID;
    bool elemCore;
    float ropeLength;
    int depth;
    int numSegments;
    float segHeight;
};

#endif // _Rope_PLUGIN_RSHARED_H
