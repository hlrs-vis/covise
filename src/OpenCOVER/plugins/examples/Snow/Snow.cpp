/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/MatrixTransform>

#include "Snow.h"
#include "Raytracer.h"
using namespace opencover;
using namespace covise;

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPlugin.h>
#include "nodeVisitorVertex.h"

#ifdef _WIN32
inline double drand48(void)
{
    return (double(rand()) / RAND_MAX);
}
#endif
SnowPlugin *SnowPlugin::plugin = NULL;

SnowPlugin::SnowPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, snow(NULL)
, x(NULL)
, y(NULL)
, z(NULL)
, r(NULL)
, nx(NULL)
, ny(NULL)
, nz(NULL)
, FloorHeight(-10.)
{
    cerr << "Let it snow" << endl;
    plugin = this;

   
}


void SnowPlugin::seed(int i)
{
    double rnd = drand48();
    rnd = pow(1.1, rnd) - 1.0;
    y[i] = MinY + drand48() * (MaxY - MinY);
    rnd = pow(1.1, drand48()) - 1.0;
    x[i] = MinX + drand48() * (MaxX - MinX);
    //x[i] = (MinX + MaxX) * 0.5 + (drand48() - 0.5) * (MaxX - MinX) * (y[i] / MaxY + 0.1) * 3;
    z[i] = StartHeight;// +(y[i] / MaxY + 0.1) * 0.01;
    r[i] = (1. + drand48()) * 0.5 * Radius;


    vx[i] = 0;// Wind * (drand48() - 0.5);
    vy[i] = 0;//Wind * (drand48() - 0.5);
    vz[i] = r[i] * Gravity;

    nx[i] = drand48() - 0.5;
    ny[i] = drand48() - 0.5;
    nz[i] = (drand48() - 0.5) * 0.5;
    isFixed[i] = 0;
}

bool SnowPlugin::init()
{
    FloorHeight = 0.0;

    Raytracer::instance()->init();

    osg::Group *scene = cover->getScene();
    osg::Matrix mat;
    mat.makeIdentity();
    //mat.setTrans(osg::Vec3(0.0,0.0,-1250.0));
    origin = new osg::MatrixTransform;
    origin->setMatrix(mat);
    cover->getObjectsRoot()->addChild(origin.get());

    osg::Geode *geode = new osg::Geode;
    origin->addChild(geode);
    snow = new coSphere();
    geode->addDrawable(snow);
    snow->setRenderMethod(coSphere::RENDER_METHOD_TEXTURE);
    x = new float[NumFlakes];
    y = new float[NumFlakes];
    z = new float[NumFlakes];
    r = new float[NumFlakes];
    vx = new float[NumFlakes];
    vy = new float[NumFlakes];
    vz = new float[NumFlakes];
    nx = new float[NumFlakes];
    ny = new float[NumFlakes];
    nz = new float[NumFlakes];
    isFixed = new int[NumFlakes];
    for (int i = 0; i < NumFlakes; ++i)
    {
        seed(i);
        if (i < NumFixFlakes || drand48() < 0.9)
        {
            z[i] = FloorHeight + 0.01;
            isFixed[i] = 1;
        }
        else
        {
            z[i] *= drand48();
            isFixed[i] = 0;
        }
    }
    snow->setCoords(NumFlakes, x, y, z, r);
    snow->updateNormals(nx, ny, nz);

    osg::StateSet *geostate = geode->getOrCreateStateSet();
    osg::Texture2D *tex = coVRFileManager::instance()->loadTexture("src/renderer/OpenCOVER/plugins/examples/Snow/Snowflake1.jpg");
    if (tex)
    {
        tex->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR_MIPMAP_LINEAR);
        tex->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
        tex->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
        tex->setWrap(osg::Texture2D::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
        geostate->setTextureAttributeAndModes(0, tex, osg::StateAttribute::ON);
    }

    //Traverse scenegraph to extract vertices for raytracer

    std::clock_t begin = clock();

    Raytracer::instance()->removeAllGeometry();                     //resets scene
    nodeVisitorVertex nv;                                            //creates new scene
    cover->getObjectsRoot()->accept(nv);
    Raytracer::instance()->createCube(osg::Vec3(0, 0, 0), osg::Vec3(2, 2, 2));
    Raytracer::instance()->createFaceSet(nv.getVertexArray(), 0);
    Raytracer::instance()->finishAddGeometry();

    std::clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    printf("elapsed time for traversing %f, vertices read out %i\n", elapsed_secs, nv.numOfVertices);

    return true;
}

bool SnowPlugin::destroy()
{
    osg::Group *scene = cover->getScene();
    scene->removeChild(origin.get());
    return true;
}

bool SnowPlugin::update()
{
    static double lastTime = 0.;
    double curTime = cover->frameTime();
    double dt = curTime - lastTime;
    if (lastTime == 0.)
        dt = 0.01;

    static double xwind = (drand48() - 0.5) * Wind;
    static double ywind = (drand48() - 0.5) * Wind;
    if (drand48() < 0.01)
    {
        xwind += (drand48() - 0.5) * Wind * 0.5;
        if (xwind < -Wind)
            xwind = -Wind;
        if (xwind > Wind)
            xwind = Wind;

        ywind += (drand48() - 0.5) * Wind * 0.5;
        if (ywind < -Wind)
            ywind = Wind;
        if (ywind > Wind)
            ywind = Wind;
    }

    if (!snow)
        return false;
    Raytracer::instance()->checkAllHits();
    for (int i = NumFixFlakes; i < NumFlakes; ++i)
    {
        if ((z[i] > FloorHeight + 0.01) && (isFixed[i]==0))
        {
           /* z[i] -= dt * (drand48() * vz[i]);
            x[i] += dt * (xwind + vx[i]);
            y[i] += dt * (ywind + vy[i]);*/
            z[i] -= dt * (vz[i]);
            x[i] += dt * (vx[i]);
            y[i] += dt * (vy[i]);
            if (drand48() < dt * 10.)
            {
                vx[i] += (drand48() - 0.5) * Wind * 0.1;
                vy[i] += (drand48() - 0.5) * Wind * 0.1;
            }

            if (drand48() < dt * 10.)
            {
                nx[i] += drand48() * 0.5;
                ny[i] += drand48() * 0.5;
                nz[i] += drand48() * 0.5;
                normalize(i);
            }
        }
        else
        {
            if (drand48() < SeedRate * dt)
                seed(i);
        }
    }
    snow->updateCoords(x, y, z);

    lastTime = curTime;
    return true;
}
SnowPlugin *SnowPlugin::instance()
{
    return plugin;
}

COVERPLUGIN(SnowPlugin);
