/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Messwerte_PLUGIN_H
#define _Messwerte_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2001 HLRS  **
**                                                                          **
** Description: Messwerte Plugin (does nothing)                              **
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
using namespace covise;
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

#include <config/CoviseConfig.h>
#include <util/byteswap.h>
#include <net/covise_connect.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>

#include <PluginUtil/colors/ColorBar.h>


class MesswertePlugin : public coVRPlugin, public coTUIListener
{
public:
    MesswertePlugin();
    ~MesswertePlugin();
    bool init();

    // this will be called in PreFrame
    void preFrame();
    void createGeom();
    void calcColors();

    int numVertTotal;
    int numFaces;
    int *numVertices;

    int numColors;
    int currentMap;
    QStringList mapNames;
    QMap<QString, int> mapSize;
    QMap<QString, float *> mapValues;

    typedef float FlColor[5];

private:
    void tabletPressEvent(coTUIElement *);
    void tabletEvent(coTUIElement *);
    void deleteColorMap(const QString &);
    void readConfig();
    void generateTexture();
    osg::Vec4 getColor(float pos);
    bool readVal(void *buf, unsigned int numBytes);

    void updateColormap();

    void readCoords(const char *filename);
    void readVertices(const char *filename, int offset = 0);
    SimpleClientConnection *conn;
    Host *serverHost;
    Host *localHost;
    int port;
    double oldTime;
    float *floatValues;
    ColorBar *cBar;

    float minVal;
    float maxVal;

    int poly;
    int ind;
    int vertNum;

    coTUITab *MesswerteTab;
    coTUIToggleButton *wireframeButton;
    coTUIToggleButton *discreteColorsButton;
    coTUILabel *mapLabel;
    coTUIComboBox *mapChoice;

    coTUILabel *minLabel;
    coTUILabel *maxLabel;
    coTUILabel *numStepsLabel;
    coTUIEditFloatField *minEdit;
    coTUIEditFloatField *maxEdit;
    coTUIEditIntField *numStepsEdit;

    osg::ref_ptr<osg::Geometry> geom;
    osg::ref_ptr<osg::Geode> geode;
    osg::ref_ptr<osg::Geometry> geomTex;
    osg::ref_ptr<osg::Geode> geodeTex;
    osg::ref_ptr<osg::Material> globalmtl;
    osg::ref_ptr<osg::DrawArrayLengths> primitives;
    osg::ref_ptr<osg::UShortArray> indices;
    osg::ref_ptr<osg::Vec3Array> vert;
    osg::ref_ptr<osg::UShortArray> nindices;
    osg::ref_ptr<osg::Vec3Array> normals;
    osg::ref_ptr<osg::UShortArray> cindices;
    osg::ref_ptr<osg::Vec4Array> colors;
    osg::ref_ptr<osg::Vec2Array> texCoords;
    osg::ref_ptr<osg::Texture2D> tex;
    osg::ref_ptr<osg::Image> texImage;
};
#endif
