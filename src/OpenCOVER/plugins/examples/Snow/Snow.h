#ifndef SNOW_PLUGIN
#define SNOW_PLUGIN

#include <config/CoviseConfig.h>
using namespace covise;

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPlugin.h>
#include <PluginUtil/coSphere.h>

class SnowPlugin : public opencover::coVRPlugin
{
public:
    static SnowPlugin *plugin;
    static const int NumFlakes = 120000;
    static const int NumFixFlakes = 80000;

    osg::ref_ptr<osg::MatrixTransform> origin;
    opencover::coSphere *snow;
    float *x, *y, *z, *r, *nx, *ny, *nz, *vx, *vy, *vz;
    float FloorHeight;

    const double Wind = 200.0;
    const double SeedRate = 0.003;
    const double Gravity = 9.81;
    const double StartHeight = 1500.;
    const float Radius = 25.0;



    const float MinX = -30000., MaxX = 30000.;
    const float MinY = -1000., MaxY = 100000.;

public:
    SnowPlugin();

    inline void normalize(int i)
    {
        float n = sqrt(nx[i] * nx[i] + ny[i] * ny[i] + nz[i] * nz[i]);
        if (n > 0.)
        {
            nx[i] /= n;
            ny[i] /= n;
            nz[i] /= n;
        }
    }

    void seed(int i);

    bool init();

    bool destroy();

    bool update();
    static SnowPlugin *instance();
};

#endif