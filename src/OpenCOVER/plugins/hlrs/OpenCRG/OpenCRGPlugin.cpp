/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2001 HLRS  **
 **                                                                          **
 ** Description: OpenCRG Plugin (does nothing)                               **
 **                                                                          **
 **                                                                          **
 ** Author: F.Seybold      	                                               **
 **                                                                          **
 ** History:  					         			                                **
 ** Nov-01  v1	    				              		                             **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include "OpenCRGPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRLighting.h>
#include <cover/VRSceneGraph.h>
#include <osgDB/ReadFile>
#include <osg/Texture2D>
#include <osg/Material>
//#include <osg/LightSource>
#include <osg/PositionAttitudeTransform>

#include <sstream>

#include <osg/Geometry>
#include <osgDB/WriteFile>

OpenCRGPlugin::OpenCRGPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, surface(NULL)
, surfaceGeode(NULL)
{
    fprintf(stderr, "OpenCRGPlugin::OpenCRGPlugin\n");
}

// this is called if the plugin is removed at runtime
OpenCRGPlugin::~OpenCRGPlugin()
{
    fprintf(stderr, "OpenCRGPlugin::~OpenCRGPlugin\n");

    delete opencrgTab;

    if (surface)
    {
        delete surface;
    }
}

void
OpenCRGPlugin::preFrame()
{
}

bool OpenCRGPlugin::init()
{
    opencrgTab = new coTUITab("OpenCRG", coVRTui::instance()->mainFolder->getID());
    opencrgTab->setPos(0, 0);

    openCrgFileButton = new coTUIFileBrowserButton("Open CRG File...", opencrgTab->getID());
    openCrgFileButton->setPos(0, 0);
    openCrgFileButton->setEventListener(this);
    openCrgFileButton->setMode(coTUIFileBrowserButton::OPEN);
    openCrgFileButton->setFilterList("*.crg");

    shadeRoadSurfaceButton = new coTUIFileBrowserButton("Shade Road Surface...", opencrgTab->getID());
    shadeRoadSurfaceButton->setPos(0, 1);
    shadeRoadSurfaceButton->setEventListener(this);
    shadeRoadSurfaceButton->setMode(coTUIFileBrowserButton::OPEN);
    shadeRoadSurfaceButton->setFilterList("*.crg");

    return (true);
}

void OpenCRGPlugin::tabletEvent(coTUIElement *tUIItem)
{
    if (tUIItem == openCrgFileButton)
    {
        std::string filename = openCrgFileButton->getSelectedPath();
        size_t spos = filename.find("file://");
        if (spos != std::string::npos)
        {
            filename.erase(spos, 7);
        }
        if (surface)
        {
            //delete surface;
            //surface = NULL;
        }
        else
        {
            std::cerr << "Opening crg file " << filename << std::endl;
            surface = new opencrg::Surface(filename);
            std::cerr << "Opened crg file: " << std::endl;
            std::cerr << (*surface) << std::endl;

            surface->computeNormals();

            double reference_line_increment = surface->getParameterValue("reference_line_increment");
            double long_section_v_increment = surface->getParameterValue("long_section_v_increment");

            if (surfaceGeode)
            {
                cover->getObjectsRoot()->removeChild(surfaceGeode);
                surfaceGeode = NULL;
            }
            surfaceGeode = new osg::Geode();

            osg::Geometry *surfaceGeometry = new osg::Geometry();
            surfaceGeode->addDrawable(surfaceGeometry);

            osg::Vec3Array *surfaceVertices = new osg::Vec3Array;
            surfaceGeometry->setVertexArray(surfaceVertices);

            osg::Vec3Array *surfaceNormals = new osg::Vec3Array;
            surfaceGeometry->setNormalArray(surfaceNormals);
            surfaceGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

            //surfaceGeode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
            osg::StateSet *surfaceGeodeState = surfaceGeode->getOrCreateStateSet();

            osg::Vec2Array *surfaceTexCoords = new osg::Vec2Array;
            surfaceGeometry->setTexCoordArray(0, surfaceTexCoords);
            osg::Texture2D *surfaceTex = new osg::Texture2D;
            //const char *texFile = coVRFileManager::instance()->getName("share/covise/materials/roadTex.jpg");
            //const char *texFile = coVRFileManager::instance()->getName("share/covise/materials/stone.png");
            const char *texFile = coVRFileManager::instance()->getName("share/covise/materials/belgianBlock.png");
            if (texFile)
            {
                osg::Image *surfaceTexImage = osgDB::readImageFile(texFile);
                surfaceTex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
                surfaceTex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
                if (surfaceTexImage)
                {
                    surfaceTex->setImage(surfaceTexImage);
                }
                surfaceGeodeState->setTextureAttributeAndModes(0, surfaceTex);
            }

            //osg::Material* surfaceGeodeMaterial = new osg::Material;
            //surfaceGeodeMaterial->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);
            //surfaceGeodeState->setAttribute(surfaceGeodeMaterial);

            //surfaceGeodeState->setMode(GL_LIGHTING, osg::StateAttribute::ON);

            unsigned int firstStripVertex = 0;
            unsigned int numStripVertices;
            for (unsigned int v = 0; v < surface->getNumberOfHeightDataElements() - 1; ++v)
            {
                numStripVertices = 0;
                for (unsigned int u = 0; u < surface->getNumberOfSurfaceDataLines(); ++u)
                {
                    float heightTwo = surface->gridPointElevation(u, v + 1) - surface->getMinimumElevation();
                    if (heightTwo == heightTwo)
                    {
                        surfaceVertices->push_back(osg::Vec3((double)u * reference_line_increment, (double)(v + 1) * long_section_v_increment, heightTwo));
                    }
                    else
                    {
                        surfaceVertices->push_back(osg::Vec3((double)u * reference_line_increment, (double)(v + 1) * long_section_v_increment, 0.0));
                    }
                    opencrg::SurfaceNormal *normal = surface->getSurfaceNormal(u, v + 1);
                    surfaceNormals->push_back(osg::Vec3(normal->u, normal->v, normal->w));
                    surfaceTexCoords->push_back(osg::Vec2((double)u * reference_line_increment, (double)(v + 1) * long_section_v_increment));
                    ++numStripVertices;

                    float heightOne = surface->gridPointElevation(u, v) - surface->getMinimumElevation();
                    if (heightOne == heightOne)
                    {
                        surfaceVertices->push_back(osg::Vec3((double)u * reference_line_increment, (double)v * long_section_v_increment, heightOne));
                    }
                    else
                    {
                        surfaceVertices->push_back(osg::Vec3((double)u * reference_line_increment, (double)v * long_section_v_increment, 0.0));
                    }
                    normal = surface->getSurfaceNormal(u, v);
                    surfaceNormals->push_back(osg::Vec3(normal->u, normal->v, normal->w));
                    surfaceTexCoords->push_back(osg::Vec2((double)u * reference_line_increment, (double)v * long_section_v_increment));
                    ++numStripVertices;
                }
                osg::DrawArrays *surfaceBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, firstStripVertex, numStripVertices);
                surfaceGeometry->addPrimitiveSet(surfaceBase);
                firstStripVertex += (numStripVertices);
            }

            /*coVRLighting::instance()->headlight->getLight()->setPosition(osg::Vec4(0.0, 100.0, 100.0, 1.0));
         coVRLighting::instance()->headlight->getLight()->setDirection(osg::Vec3(0.0, -1.0, -1.0));
         coVRLighting::instance()->headlight->getLight()->setDiffuse(osg::Vec4(1.0, 1.0, 1.0, 1.0));
         coVRLighting::instance()->headlight->getLight()->setAmbient(osg::Vec4(0.7, 0.7, 0.7, 0.7));
         coVRLighting::instance()->headlight->getLight()->setSpecular(osg::Vec4(0.1, 0.1, 0.1, 0.1));*/

            osg::PositionAttitudeTransform *surfaceTransform = new osg::PositionAttitudeTransform();
            surfaceTransform->setName("tesselated");
            surfaceTransform->setPosition(osg::Vec3d(0.0, 0.0, 0.0));
            //surfaceTransform->setScale(osg::Vec3d(1000.0, 1000.0, 1000.0));
            surfaceTransform->addChild(surfaceGeode);
            cover->getObjectsRoot()->addChild(surfaceTransform);

            osg::PositionAttitudeTransform *bumpTransform = new osg::PositionAttitudeTransform();
            bumpTransform->setName("bumped");
            bumpTransform->setPosition(osg::Vec3d(0.0, 10.0, 0.0));
            cover->getObjectsRoot()->addChild(bumpTransform);

            osg::Geode *bumpGeode = new osg::Geode;
            bumpTransform->addChild(bumpGeode);

            osg::Geometry *bumpGeometry = new osg::Geometry;
            bumpGeode->addDrawable(bumpGeometry);

            osg::Vec3Array *bumpVertices = new osg::Vec3Array;
            bumpGeometry->setVertexArray(bumpVertices);

            osg::Vec3Array *bumpNormals;
            bumpNormals = new osg::Vec3Array;
            bumpGeometry->setNormalArray(bumpNormals);
            bumpGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

            osg::Vec3Array *bumpTangents = new osg::Vec3Array;
            bumpGeometry->setVertexAttribArray(6, bumpTangents);
            bumpGeometry->setVertexAttribBinding(6, osg::Geometry::BIND_PER_VERTEX);

            osg::Vec3Array *bumpBinormals = new osg::Vec3Array;
            bumpGeometry->setVertexAttribArray(7, bumpBinormals);
            bumpGeometry->setVertexAttribBinding(7, osg::Geometry::BIND_PER_VERTEX);

            osg::Vec2Array *bumpTexCoords = new osg::Vec2Array;
            bumpGeometry->setTexCoordArray(0, bumpTexCoords);
            bumpGeometry->setTexCoordArray(1, bumpTexCoords);

            bumpGeode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
            osg::StateSet *bumpGeodeState = bumpGeode->getOrCreateStateSet();
            if (texFile)
            {
                bumpGeodeState->setTextureAttributeAndModes(0, surfaceTex);
            }
            osg::DrawArrays *bumpBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, 0, 4);
            bumpGeometry->addPrimitiveSet(bumpBase);
            double bumpLength = surface->getLength();
            double bumpWidth = surface->getWidth();
            bumpVertices->push_back(osg::Vec3(0.0, 0.0, 0.0));
            bumpTexCoords->push_back(osg::Vec2(0.0, 0.0));
            bumpVertices->push_back(osg::Vec3(0.0, bumpWidth, 0.0));
            bumpTexCoords->push_back(osg::Vec2(0.0, 1.0));
            bumpVertices->push_back(osg::Vec3(bumpLength, 0.0, 0.0));
            bumpTexCoords->push_back(osg::Vec2(1.0, 0.0));
            bumpVertices->push_back(osg::Vec3(bumpLength, bumpWidth, 0.0));
            bumpTexCoords->push_back(osg::Vec2(1.0, 1.0));
            bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
            bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
            bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
            bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
            bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
            bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
            bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
            bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
            bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));
            bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));
            bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));
            bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));

            osg::Texture2D *parallaxMapTexture = new osg::Texture2D;
            parallaxMapTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
            parallaxMapTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
            parallaxMapTexture->setImage(surface->createParallaxMapTextureImage());
            //parallaxMapTexture->setImage(osgDB::readImageFile("/media/Data/data/shader/parallax_mapping/parallaxMap.png"));
            bumpGeodeState->setTextureAttributeAndModes(1, parallaxMapTexture);

            //osg::Image* parallaxMapImage = surface->createParallaxMapTextureImage();
            //osgDB::writeImageFile(*parallaxMapImage, "roadParallaxMap.bmp");

            coVRShader *parallaxShader = coVRShaderList::instance()->get("parallaxMapping");
            if (parallaxShader == NULL)
            {
                cerr << "ERROR: no shader found with name: parallaxMapping" << endl;
            }
            else
            {
                parallaxShader->apply(bumpGeode);
            }

            //Cone Step Mapping
            osg::PositionAttitudeTransform *csmTransform = new osg::PositionAttitudeTransform();
            csmTransform->setName("csmed");
            csmTransform->setPosition(osg::Vec3d(0.0, -10.0, 0.0));
            cover->getObjectsRoot()->addChild(csmTransform);

            osg::Geode *csmGeode = new osg::Geode;
            csmTransform->addChild(csmGeode);

            csmGeode->addDrawable(bumpGeometry);

            csmGeode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
            osg::StateSet *csmGeodeState = csmGeode->getOrCreateStateSet();
            if (texFile)
            {
                csmGeodeState->setTextureAttributeAndModes(0, surfaceTex);
            }

            osg::Texture2D *csmMapTexture = new osg::Texture2D;
            csmMapTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
            csmMapTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
            csmMapTexture->setImage(surface->createConeStepMapTextureImage());
            csmGeodeState->setTextureAttributeAndModes(1, csmMapTexture);

            coVRShader *csmShader = coVRShaderList::instance()->get("coneStepMapping");
            if (csmShader == NULL)
            {
                cerr << "ERROR: no shader found with name: csmMapping" << endl;
            }
            else
            {
                csmShader->apply(csmGeode);
            }
        }
    }
    else if (tUIItem == shadeRoadSurfaceButton)
    {
        std::string filename = shadeRoadSurfaceButton->getSelectedPath();
        size_t spos = filename.find("file://");
        if (spos != std::string::npos)
        {
            filename.erase(spos, 7);
        }

        if (!surface)
        {
            std::cerr << "Opening crg file " << filename << std::endl;
            surface = new opencrg::Surface(filename);
            std::cerr << "Opened crg file: " << std::endl;
            std::cerr << (*surface) << std::endl;

            surface->computeNormals();

            double reference_line_increment = surface->getParameterValue("reference_line_increment");
            double long_section_v_increment = surface->getParameterValue("long_section_v_increment");

            coVRLighting::instance()->headlight->getLight()->setPosition(osg::Vec4(0.0, 100.0, 100.0, 1.0));
            coVRLighting::instance()->headlight->getLight()->setDirection(osg::Vec3(0.0, -1.0, -1.0));
            coVRLighting::instance()->headlight->getLight()->setDiffuse(osg::Vec4(1.0, 1.0, 1.0, 1.0));
            coVRLighting::instance()->headlight->getLight()->setAmbient(osg::Vec4(0.7, 0.7, 0.7, 0.7));
            coVRLighting::instance()->headlight->getLight()->setSpecular(osg::Vec4(0.1, 0.1, 0.1, 0.1));

            osg::PositionAttitudeTransform *bumpTransform = new osg::PositionAttitudeTransform();
            bumpTransform->setName("bumped");
            bumpTransform->setPosition(osg::Vec3d(0.0, 10.0, 0.0));
            cover->getObjectsRoot()->addChild(bumpTransform);

            osg::Geode *bumpGeode = new osg::Geode;
            bumpTransform->addChild(bumpGeode);

            osg::Geometry *bumpGeometry = new osg::Geometry;
            bumpGeode->addDrawable(bumpGeometry);

            osg::Vec3Array *bumpVertices = new osg::Vec3Array;
            bumpGeometry->setVertexArray(bumpVertices);

            osg::Vec3Array *bumpNormals;
            bumpNormals = new osg::Vec3Array;
            bumpGeometry->setNormalArray(bumpNormals);
            bumpGeometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

            osg::Vec3Array *bumpTangents = new osg::Vec3Array;
            bumpGeometry->setVertexAttribArray(6, bumpTangents);
            bumpGeometry->setVertexAttribBinding(6, osg::Geometry::BIND_PER_VERTEX);

            osg::Vec3Array *bumpBinormals = new osg::Vec3Array;
            bumpGeometry->setVertexAttribArray(7, bumpBinormals);
            bumpGeometry->setVertexAttribBinding(7, osg::Geometry::BIND_PER_VERTEX);

            osg::Vec2Array *bumpTexCoords = new osg::Vec2Array;
            bumpGeometry->setTexCoordArray(0, bumpTexCoords);
            bumpGeometry->setTexCoordArray(1, bumpTexCoords);

            osg::DrawArrays *bumpBase = new osg::DrawArrays(osg::PrimitiveSet::TRIANGLE_STRIP, 0, 4);
            bumpGeometry->addPrimitiveSet(bumpBase);
            double bumpLength = surface->getNumberOfSurfaceDataLines() * reference_line_increment;
            double bumpWidth = surface->getNumberOfHeightDataElements() * long_section_v_increment;
            bumpVertices->push_back(osg::Vec3(0.0, 0.0, 0.0));
            bumpTexCoords->push_back(osg::Vec2(0.0, 0.0));
            bumpVertices->push_back(osg::Vec3(0.0, bumpWidth, 0.0));
            bumpTexCoords->push_back(osg::Vec2(0.0, 1.0));
            bumpVertices->push_back(osg::Vec3(bumpLength, 0.0, 0.0));
            bumpTexCoords->push_back(osg::Vec2(1.0, 0.0));
            bumpVertices->push_back(osg::Vec3(bumpLength, bumpWidth, 0.0));
            bumpTexCoords->push_back(osg::Vec2(1.0, 1.0));
            bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
            bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
            bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
            bumpNormals->push_back(osg::Vec3(0.0, 0.0, 1.0));
            bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
            bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
            bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
            bumpTangents->push_back(osg::Vec3(1.0, 0.0, 0.0));
            bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));
            bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));
            bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));
            bumpBinormals->push_back(osg::Vec3(0.0, 1.0, 0.0));

            bumpGeode->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
            osg::StateSet *bumpGeodeState = bumpGeode->getOrCreateStateSet();

            osg::Texture2D *diffuseMapTexture = new osg::Texture2D;
            diffuseMapTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
            diffuseMapTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
            diffuseMapTexture->setImage(surface->createDiffuseMapTextureImage());
            bumpGeodeState->setTextureAttributeAndModes(0, diffuseMapTexture);

            osg::Texture2D *parallaxMapTexture = new osg::Texture2D;
            parallaxMapTexture->setWrap(osg::Texture2D::WRAP_S, osg::Texture::REPEAT);
            parallaxMapTexture->setWrap(osg::Texture2D::WRAP_T, osg::Texture::REPEAT);
            parallaxMapTexture->setImage(surface->createParallaxMapTextureImage());
            bumpGeodeState->setTextureAttributeAndModes(1, parallaxMapTexture);

            coVRShader *parallaxShader = coVRShaderList::instance()->get("parallaxMapping");
            if (parallaxShader == NULL)
            {
                cerr << "ERROR: no shader found with name: parallaxMapping" << endl;
            }
            else
            {
                parallaxShader->apply(bumpGeode);
            }
        }
    }
}

COVERPLUGIN(OpenCRGPlugin)
