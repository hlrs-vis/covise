/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Local:
#include "PointCloud.h"
#include "PointCloudGeometry.h"

#include <iostream>
#include <osg/Point>
#include <osg/Image>
#include <osg/AlphaFunc>
#include <osg/TexEnv>
#include <osg/TexGen>
#include <cover/coVRFileManager.h>

using namespace osg;
using namespace std;

PointCloudGeometry::PointCloudGeometry()
{
}

// should be using a file pointer instead of loading all data into memory
PointCloudGeometry::PointCloudGeometry(PointSet *pointData)
{
    // load data outside of the drawable
    setUseDisplayList(false);
    setSupportsDisplayList(false);
    setUseVertexBufferObjects(true);

    // maximum point size
    maxPointSize = 3.0;

    // save copy of pointData pointer
    pointSet = pointData;

    // create buffer objects to hold point data
    vertexBufferArray = getOrCreateVertexBufferObject();
    //vertexBufferArray->setUsage(GL_STREAM_DRAW_ARB);
    //primitiveBufferArray = getOrCreateElementBufferObject();

    // set color and vertexArrays
    colors = new Vec3Array(pointSet->size, (osg::Vec3 *)pointSet->colors);
    points = new Vec3Array(pointSet->size, (osg::Vec3 *)pointSet->points);

    vertexBufferArray->setArray(0, points);
    vertexBufferArray->setArray(1, colors);
    // bind color per vertex
    points->setBinding(osg::Array::BIND_PER_VERTEX);
    colors->setBinding(osg::Array::BIND_PER_VERTEX);

    setVertexArray(points);
    setColorArray(colors);

    // default initalization (modes 1,2,4,8)
    subsample = 0.3;

    // init bounding box
    box.init();

    // after test move stateset higher up in the tree
    stateset = new StateSet();
    //stateset->setMode(GL_PROGRAM_POINT_SIZE_EXT, StateAttribute::ON);
    stateset->setMode(GL_LIGHTING, StateAttribute::OFF);
    stateset->setMode(GL_DEPTH_TEST, StateAttribute::ON);
    stateset->setMode(GL_ALPHA_TEST, StateAttribute::ON);
    stateset->setMode(GL_BLEND, StateAttribute::OFF);
    AlphaFunc *alphaFunc = new AlphaFunc(AlphaFunc::GREATER, 0.5);
    stateset->setAttributeAndModes(alphaFunc, StateAttribute::ON);

    pointstate = new osg::Point();
    pointstate->setSize(PointCloudPlugin::plugin->pointSize());
    stateset->setAttributeAndModes(pointstate, StateAttribute::ON);

    osg::PointSprite *sprite = new osg::PointSprite();
    if (PointCloudPlugin::plugin->usePointSprites())
    {
        stateset->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::ON);

        const char* mapName = opencover::coVRFileManager::instance()->getName("share/covise/icons/particle.png");
        if (mapName != NULL)
        {
            osg::Image* image = osgDB::readImageFile(mapName);
            osg::Texture2D* tex = new osg::Texture2D(image);

            tex->setTextureSize(image->s(), image->t());
            tex->setInternalFormat(GL_RGBA);
            tex->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
            tex->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
            stateset->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
            osg::TexEnv* texEnv = new osg::TexEnv;
            texEnv->setMode(osg::TexEnv::MODULATE);
            stateset->setTextureAttributeAndModes(0, texEnv, osg::StateAttribute::ON);

            osg::ref_ptr<osg::TexGen> texGen = new osg::TexGen();
            stateset->setTextureAttributeAndModes(0, texGen.get(), osg::StateAttribute::OFF);

        }
    }
    else
    {
        stateset->setTextureAttributeAndModes(0, sprite, osg::StateAttribute::OFF);
    }

    /*osg::Program* program = new osg::Program;
        stateset->setAttribute(program);

        char vertexShaderSource[] = 
            "\n"
            "\n"
            "varying vec4 colour;\n"
            "\n"
            "void main(void)\n"
            "{\n"
            "    float startTime = gl_MultiTexCoord1.x;\n"
            "\n"
            "    vec4 v_current = gl_Vertex;\n"
            "    colour = gl_Color;\n"
            "\n"
            "    gl_Position = gl_ModelViewProjectionMatrix * v_current;\n"
            "\n"
            "    //float pointSize = abs(1280.0*particleSize / gl_Position.w);\n"
            "\n"
            "    //gl_PointSize = max(ceil(pointSize),2);\n"
            "    //gl_PointSize = ceil(pointSize);\n"
            "    \n"
            "    //colour.a = gl_PointSize;\n"
            "    gl_PointSize = 8;\n"
            "    gl_ClipVertex = v_current;\n"
            "}\n";

        char fragmentShaderSource[] = 
            "uniform sampler2D baseTexture;\n"
            "varying vec4 colour;\n"
            "\n"
            "void main (void)\n"
            "{\n"
            "    gl_FragColor = colour * texture2D( baseTexture, gl_TexCoord[0].xy);\n"
            "}\n";

        program->addShader(new osg::Shader(osg::Shader::VERTEX, vertexShaderSource));
        program->addShader(new osg::Shader(osg::Shader::FRAGMENT, fragmentShaderSource));
            stateset->addUniform(new osg::Uniform("baseTexture",0));
            
                  stateset->setRenderBinDetails(-1,"RenderBin");
stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF); */

    setStateSet(stateset);

    updateBounds();
}


void PointCloudGeometry::updateBounds()
{
    // init bounding box
    box.init();

    //expand box
    ::Point *data = pointSet->points;

    for (int i = 0; i < pointSet->size; i++)
    {
        box.expandBy(data[i].coordinates.x(), data[i].coordinates.y(), data[i].coordinates.z());
    }
    setInitialBound(box);
}

PointCloudGeometry::PointCloudGeometry(const PointCloudGeometry &drawimage, const CopyOp &copyop)
    : Geometry(drawimage, copyop)
{
    // make copies of global variables here; optional
}

PointCloudGeometry::~PointCloudGeometry()
{
    //cerr << "~PointCloudPluginDrawable() called!\n" << endl;
}

#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 2)
BoundingBox PointCloudGeometry::computeBoundingBox() const
#else
BoundingBox PointCloudGeometry::computeBound() const
#endif
{
    return box;
}

// need to recode in a better way to automatically adjust primitives based on depth  //TODO
void PointCloudGeometry::changeLod(float sampleNum)
{
    if (sampleNum < 0)
    {
        sampleNum = 0.;
    }
    if (sampleNum > 1.)
    {
        sampleNum = 1.;
    }

    if (subsample == sampleNum)
    {
        return;
    }

    subsample = sampleNum;

    if (getNumPrimitiveSets() == 0)
    {
        addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POINTS, 0, (int)(pointSet->size * sampleNum)));
    }
    else
    {
        auto prim = getPrimitiveSet(0);
        auto arr = static_cast<osg::DrawArrays *>(prim);
        arr->setCount(pointSet->size * sampleNum);
    }

    setPointSize(pointSize);
}

void PointCloudGeometry::setPointSize(float newPointSize)
{
    pointSize = newPointSize;
    pointstate->setSize(pointSize / ((subsample / 4.0) + (3.0 / 4.0)));
}

void PointCloudGeometry::updateCoords()
{
    memcpy(&points->at(0), pointSet->points, pointSet->size*sizeof(pointSet->points[0]));
    points->dirty();
    updateBounds();
}
