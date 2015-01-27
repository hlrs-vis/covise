/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osg/MatrixTransform>

#include <config/CoviseConfig.h>
using namespace covise;

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPlugin.h>
#include <PluginUtil/coSphere.h>
using namespace opencover;

#ifdef _WIN32
inline double drand48(void)
{
    return (double(rand()) / RAND_MAX);
}
#endif

class SnowPlugin : public coVRPlugin
{
private:
    static const int NumFlakes = 120000;
    static const int NumFixFlakes = 80000;

    osg::ref_ptr<osg::MatrixTransform> origin;
    coSphere *snow;
    float *x, *y, *z, *r, *nx, *ny, *nz, *vx, *vy;
    float FloorHeight;

    static const double Wind;
    static const double SeedRate;
    static const double Gravity;
    static const double StartHeight;
    static const float Radius;

    static const float MinX, MaxX;
    static const float MinY, MaxY;

public:
    SnowPlugin()
        : snow(NULL)
        , x(NULL)
        , y(NULL)
        , z(NULL)
        , r(NULL)
        , nx(NULL)
        , ny(NULL)
        , nz(NULL)
        , FloorHeight(-1100.)
    {
        cerr << "Let it snow" << endl;
    }

    void normalize(int i)
    {
        float n = sqrt(nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i]);
        if (n > 0.)
        {
            nx[i] /= n;
            ny[i] /= n;
            nz[i] /= n;
        }
    }

    void seed(int i)
    {
        double rnd = drand48();
        rnd = pow(1.1, rnd) - 1.0;
        y[i] = MinY + rnd * (MaxY - MinY);
        x[i] = (MinX + MaxX) * 0.5 + (drand48() - 0.5) * (MaxX - MinX) * (y[i] / MaxY + 0.1) * 3;
        z[i] = StartHeight * (y[i] / MaxY + 0.1) * 10.;
        r[i] = (1. + drand48()) * 0.5 * Radius;

        vx[i] = Wind * (drand48() - 0.5);
        vy[i] = Wind * (drand48() - 0.5);

        nx[i] = drand48() - 0.5;
        ny[i] = drand48() - 0.5;
        nz[i] = (drand48() - 0.5) * 0.5;
    }

    bool init()
    {
        FloorHeight = coCoviseConfig::getFloat("COVER.FloorHeight", -1100);

        osg::Group *scene = cover->getScene();
        osg::Matrix mat;
        mat.makeIdentity();
        //mat.setTrans(osg::Vec3(0.0,0.0,-1250.0));
        origin = new osg::MatrixTransform;
        origin->setMatrix(mat);
        scene->addChild(origin.get());

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
        nx = new float[NumFlakes];
        ny = new float[NumFlakes];
        nz = new float[NumFlakes];
        for (int i = 0; i < NumFlakes; ++i)
        {
            seed(i);
            if (i < NumFixFlakes || drand48() < 0.9)
                z[i] = FloorHeight + 20.0;
            else
                z[i] *= drand48();
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
        return true;
    }

    bool destroy()
    {
        osg::Group *scene = cover->getScene();
        scene->removeChild(origin.get());
        return true;
    }

    void preFrame()
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
            return;

        for (int i = NumFixFlakes; i < NumFlakes; ++i)
        {
            if (z[i] > FloorHeight + 20.0)
            {
                z[i] -= dt * drand48() * r[i] * Gravity;
                x[i] += dt * (xwind + vx[i]);
                y[i] += dt * (ywind + vy[i]);
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
    }
};

const double SnowPlugin::Wind = 200.0;
const double SnowPlugin::SeedRate = 0.003;
const double SnowPlugin::Gravity = 10.0;
const double SnowPlugin::StartHeight = 1500.;
const float SnowPlugin::Radius = 25.0;

const float SnowPlugin::MinX = -30000.;
const float SnowPlugin::MaxX = 30000.;
const float SnowPlugin::MinY = -1000.;
const float SnowPlugin::MaxY = 100000.;

COVERPLUGIN(SnowPlugin);
