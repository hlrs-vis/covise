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
    //static const int NumFlakes = 120000;
    //static const int NumFixFlakes = 80000;
    static const int NumFlakes = 12000;
    static const int NumFixFlakes = 800;

    osg::ref_ptr<osg::MatrixTransform> origin;
    opencover::coSphere *snow;
    float *x, *y, *z, *r, *nx, *ny, *nz, *vx, *vy, *vz;
    int *isFixed;
    float FloorHeight;

    const double Wind = 0.20;
    const double SeedRate = 0.003;
    const double Gravity = 9.81;
    const double StartHeight = 3.;
    const float Radius = 0.0250;



    const float MinX = -5., MaxX = 5.;
    const float MinY = -5., MaxY = 5.;

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