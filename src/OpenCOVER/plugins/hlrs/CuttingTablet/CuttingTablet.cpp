/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdio.h>

#include <osg/Vec3f>
#include <osg/Geometry>
#include <osg/AlphaFunc>
#include <osg/BlendFunc>

#include <osg/ClipNode>
#include <osg/ClipPlane>
#include <osg/Array>
#include <osg/Light>
#include <osg/LightModel>
#include <osg/LightSource>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/PositionAttitudeTransform>
#include <osg/Projection>
#include <osg/Shader>
#include <osg/Texture1D>
#include <osg/Texture2D>

#include <osgDB/ReadFile>

#include <config/CoviseConfig.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>

#include "slicer.h"
#include "networkzmq.h"

#include "CuttingTablet.h"

CuttingTablet::CuttingTablet()
: coVRPlugin(COVER_PLUGIN_NAME)
, server(NULL)
, slicer(NULL)
, geode(NULL)
, root(NULL)
, count(0)
, t0(NULL)
, renderState(false)
{

    printf("::CuttingTablet\n");
}

CuttingTablet::~CuttingTablet()
{
}

bool CuttingTablet::init()
{

    float id = covise::coCoviseConfig::getInt("id", "COVER.Plugin.CuttingTablet.Marker", 0);
    float x = covise::coCoviseConfig::getFloat("x", "COVER.Plugin.CuttingTablet.Marker", 0);
    float y = covise::coCoviseConfig::getFloat("y", "COVER.Plugin.CuttingTablet.Marker", 0);
    float z = covise::coCoviseConfig::getFloat("z", "COVER.Plugin.CuttingTablet.Marker", 0);

    float rx = covise::coCoviseConfig::getFloat("rx", "COVER.Plugin.CuttingTablet.Marker", 0);
    float ry = covise::coCoviseConfig::getFloat("ry", "COVER.Plugin.CuttingTablet.Marker", 0);
    float rz = covise::coCoviseConfig::getFloat("rz", "COVER.Plugin.CuttingTablet.Marker", 0);

    rx = M_PI / 180.0 * rx;
    ry = M_PI / 180.0 * ry;
    rz = M_PI / 180.0 * rz;

    osg::Matrix mPos = osg::Matrix::translate(x, y, z);
    mPos = mPos * osg::Matrix::rotate(rx, 1.0, 0.0, 0.0);
    mPos = mPos * osg::Matrix::rotate(ry, 0.0, 1.0, 0.0);
    mPos = mPos * osg::Matrix::rotate(rz, 0.0, 0.0, 1.0);

    if (opencover::coVRMSController::instance()->isMaster())
    {
        server = new Server(42424, 42425);
        server->setMarkerMatrix(id, mPos);
    }

    slicer = new Slicer("/data/ivk/A8/A8bsmall.vtk",
                        "/data/ivk/A8/A8csmall.vtk",
                        "/data/ivk/A8/colormap.bmp",
                        "/data/ivk/A8/colormap2.bmp");
    slicer->setPlane(osg::Vec3(0.0, 0.0, 1.0), osg::Vec3(0.0, 0.0, 0.0));

    root = opencover::cover->getObjectsRoot();

    osg::ref_ptr<osg::Image> logoImage = osgDB::readImageFile("/data/ivk/A8/marker.png");

    osg::ref_ptr<osg::Geode> logo = new osg::Geode();

    osg::ref_ptr<osg::Texture2D> logoTexture = new osg::Texture2D(logoImage.get());
    //logoTexture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR_MIPMAP_LINEAR);
    //logoTexture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    logoTexture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
    logoTexture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);

    osg::ref_ptr<osg::StateSet> stateSet = new osg::StateSet();

    stateSet->setTextureAttributeAndModes(0, logoTexture.get(),
                                          osg::StateAttribute::ON);

    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::ref_ptr<osg::BlendFunc> blend = new osg::BlendFunc();
    stateSet->setAttributeAndModes(blend.get(), osg::StateAttribute::ON);

    osg::ref_ptr<osg::AlphaFunc> alpha = new osg::AlphaFunc();
    alpha->setFunction(osg::AlphaFunc::GEQUAL, 0.05);
    stateSet->setAttributeAndModes(alpha.get(), osg::StateAttribute::ON);
    stateSet->setRenderBinDetails(11, "RenderBin");
    stateSet->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateSet->setMode(GL_DEPTH_TEST, osg::StateAttribute::OFF);

    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();

    float size = 384.0;
    float aspect = 1.0;
    float originX = (1920.0 - size) / 2.0;
    float originY = 0.0;

    vertices->push_back(osg::Vec3(originX, originY, 0.0));
    vertices->push_back(osg::Vec3(originX + size, originY, 0.0));
    vertices->push_back(osg::Vec3(originX + size, originY + size * aspect, 0.0));
    vertices->push_back(osg::Vec3(originX, originY + size * aspect, 0.0));

    osg::ref_ptr<osg::DrawElementsUInt> indices = new osg::DrawElementsUInt(osg::PrimitiveSet::POLYGON, 0);
    indices->push_back(0);
    indices->push_back(1);
    indices->push_back(2);
    indices->push_back(3);

    osg::ref_ptr<osg::Vec2Array> texCoords = new osg::Vec2Array();
    texCoords->push_back(osg::Vec2(0.0, 0.0));
    texCoords->push_back(osg::Vec2(1.0, 0.0));
    texCoords->push_back(osg::Vec2(1.0, 1.0));
    texCoords->push_back(osg::Vec2(0.0, 1.0));

    geometry->setVertexArray(vertices.get());
    geometry->addPrimitiveSet(indices.get());
    geometry->setTexCoordArray(0, texCoords.get());
    geometry->setStateSet(stateSet.get());

    logo->addDrawable(geometry.get());

    osg::ref_ptr<osg::Projection> logoProj = new osg::Projection();
    logoProj->setMatrix(osg::Matrix::ortho2D(0, 1920, 0, 1080));

    osg::ref_ptr<osg::MatrixTransform> logoView = new osg::MatrixTransform();
    logoView->setMatrix(osg::Matrix::identity());
    logoView->setReferenceFrame(osg::Transform::ABSOLUTE_RF);

    logoView->addChild(logo.get());
    logoProj->addChild(logoView.get());

    if (opencover::coVRMSController::instance()->getID() == 5)
        root->addChild(logoProj.get());

    return true;
}

void CuttingTablet::preFrame()
{

    if (slicer)
    {

        osg::Vec3 normal;
        osg::Vec3 position;

        if (opencover::coVRMSController::instance()->isMaster())
        {

            int dataSet = server->getDataSet();
            server->poll();
            normal = server->getNormal();
            position = server->getPosition();
            slicer->setDataSet(dataSet);

            int r = renderState ? 1 : 0;
            opencover::coVRMSController::instance()->sendSlaves((char *)normal._v, sizeof(float) * 3);
            opencover::coVRMSController::instance()->sendSlaves((char *)position._v, sizeof(float) * 3);
            opencover::coVRMSController::instance()->sendSlaves((char *)&dataSet, sizeof(int));
            opencover::coVRMSController::instance()->sendSlaves((char *)&r, sizeof(int));
        }
        else
        {
            int dataSet, r;
            opencover::coVRMSController::instance()->readMaster((char *)normal._v, sizeof(float) * 3);
            opencover::coVRMSController::instance()->readMaster((char *)position._v, sizeof(float) * 3);
            opencover::coVRMSController::instance()->readMaster((char *)&dataSet, sizeof(int));
            opencover::coVRMSController::instance()->readMaster((char *)&r, sizeof(int));
            renderState = r;
            slicer->setDataSet(dataSet);
        }

        slicer->setPlane(normal, position);

        if (geode)
            root->removeChild(geode);

        geode = slicer->getGeode();

        if (geode != NULL && opencover::coVRMSController::instance()->isMaster())
        {

            const osg::Geometry *geom = (osg::Geometry *)geode->getDrawable(0);
            const osg::DrawElementsUInt *primitives = (osg::DrawElementsUInt *)geom->getPrimitiveSet(0);
            const osg::Vec3Array *vertices = (osg::Vec3Array *)geom->getVertexArray();
            const osg::FloatArray *texCoords = (osg::FloatArray *)geom->getTexCoordArray(0);

            if (primitives->getNumIndices() && oldPosition != position && oldNormal != normal)
            {

                //count ++;

                if (count == 20)
                {
                    count = 0;
                    oldNormal = normal;
                    oldPosition = position;

                    server->sendGeometry(vertices->getNumElements(),
                                         (float *)vertices->getDataPointer(),
                                         primitives->getNumIndices(),
                                         (unsigned int *)primitives->getDataPointer(),
                                         texCoords->getNumElements(),
                                         (float *)texCoords->getDataPointer());
                }
            }
        }
        if (renderState)
            root->addChild(geode);
    }
}

void CuttingTablet::key(int type, int keySym, int mod)
{

    switch (type)
    {
    case (osgGA::GUIEventAdapter::KEYDOWN):

        if (opencover::coVRMSController::instance()->getID() == 0)
        {

            if (keySym == 'o' || keySym == 'O')
            {

                renderState = !renderState;
            }

            if (keySym == 'p' || keySym == 'P')
            {

                if (t0 == NULL)
                {

                    t0 = (struct timeval *)malloc(sizeof(struct timeval));
                    gettimeofday(t0, NULL);
                }
                else
                {

                    gettimeofday(&t1, NULL);
                    long long t = (t1.tv_sec - t0->tv_sec);
                    t = t * 1000000;
                    t += t1.tv_usec;
                    t -= t0->tv_usec;
                    free(t0);
                    t0 = NULL;
                    printf("              %d msec\n", t / 1000);
                }
            }

            if (keySym == 'n' || keySym == 'N')
            {

                osg::Vec3 normal = server->getNormal();
                printf("  (%2.5f, %2.5f, %2.5f)\n", normal.x(), normal.y(), normal.z());
            }
        }
    }
}

COVERPLUGIN(CuttingTablet)
