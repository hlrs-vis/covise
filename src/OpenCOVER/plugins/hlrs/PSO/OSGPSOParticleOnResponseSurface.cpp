/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OSGPSOParticleOnResponseSurface.h"

#include <osg/ShapeDrawable>
#include <string.h>

#ifdef _MSC_VER
#include <ymath.h>
#ifndef INFINITY
#define INFINITY _FInf._Double
#endif
#if (_MSC_VER < 1900)
int round(double d) { return ((int)(d + 0.5)); }
int lround(double d) { return ((int)(d + 0.5)); }
#endif
#endif

std::vector<OSGPSOParticleOnResponseSurface *> OSGPSOParticleOnResponseSurface::par;
double OSGPSOParticleOnResponseSurface::t = 0;
osg::Geode *OSGPSOParticleOnResponseSurface::ParticleGeode = new osg::Geode();

OSGPSOParticleOnResponseSurface::OSGPSOParticleOnResponseSurface()
    : osg::PositionAttitudeTransform()
    , pso::Particle()
{
    par.push_back(this);

    oldX = new double[nvar];
    oldV = new double[nvar];
    memcpy(oldX, x, nvar * sizeof(double));
    memcpy(oldV, v, nvar * sizeof(double));

    p = new double *[nvar];
    for (int j = 0; j < nvar; ++j)
    {
        p[j] = new double[4];
        for (int i = 0; i < 4; ++i)
            p[j][i] = 0;
    }

    this->addChild(ParticleGeode);
}

void OSGPSOParticleOnResponseSurface::computePath()
{
    for (int j = 0; j < nvar; ++j)
    {
        p[j][0] = (-2 * x[j] + 2 * oldX[j] + dt * (v[j] + oldV[j])) / (pow(dt, 3));
        p[j][1] = -(-3 * x[j] + 3 * oldX[j] + dt * (v[j] + 2 * oldV[j])) / (pow(dt, 2));
        p[j][2] = oldV[j];
        p[j][3] = oldX[j];
    }
}

void OSGPSOParticleOnResponseSurface::move()
{
    double x[2];
    x[0] = p[0][0] * pow(t, 3) + p[0][1] * pow(t, 2) + p[0][2] * t + p[0][3];
    x[1] = p[1][0] * pow(t, 3) + p[1][1] * pow(t, 2) + p[1][2] * t + p[1][3];
    osg::Vec3 OSGPos(x[0], x[1], (*response)(x));
    setPosition(OSGPos);

    double xdir = 3 * p[0][0] * pow(t, 2) + 2 * p[0][1] * t + p[0][2];
    double ydir = 3 * p[1][0] * pow(t, 2) + 2 * p[1][1] * t + p[1][2];
    if ((xdir + ydir) > 0.000001)
    {
        osg::Quat OSGRot;
        OSGRot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(xdir, ydir, 0));
        setAttitude(OSGRot);
    }
}

void OSGPSOParticleOnResponseSurface::updateVelocity()
{
    memcpy(oldV, v, nvar * sizeof(double));
    pso::Particle::updateVelocity();
}

void OSGPSOParticleOnResponseSurface::updatePosition()
{
    memcpy(oldX, x, nvar * sizeof(double));
    pso::Particle::updatePosition();
}

void OSGPSOParticleOnResponseSurface::enforceConstrains()
{
    for (int j = 0; j < nvar; ++j)
    {
        // Rounding if integer
        if (integer[j])
            x[j] = round(x[j]);

        // Fixing bounds
        if (x[j] < lowerbound[j])
        {
            x[j] = lowerbound[j];
            v[j] = 0;
            oldV[j] = 0.0;
        }
        else if (x[j] > upperbound[j])
        {
            x[j] = upperbound[j];
            v[j] = 0;
            oldV[j] = 0.0;
        }
    }
}

void OSGPSOParticleOnResponseSurface::init(double (*setresponse)(double *), double *setlowerbound, double *setupperbound, bool *setinteger, long int seed)
{
    pso::Particle::init(setresponse, 2, setlowerbound, setupperbound, setinteger, 1);

    initRand(seed);
    par.clear();

    //ParticleGeode->addDrawable(new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(0,0,0), 10)));

    double scaleFactor = sqrt(pow(setupperbound[0] - setlowerbound[0], 2) + pow(setupperbound[1] - setlowerbound[1], 2)) / 100;
    scaleFactor = 0.01;
    ParticleGeode = new osg::Geode();
    ParticleGeode->addDrawable(new osg::ShapeDrawable(new osg::Cone(osg::Vec3(0, 0, 0), 1 * scaleFactor, 5 * scaleFactor)));
}

void OSGPSOParticleOnResponseSurface::destroy()
{
    pso::Particle::destroy();
}

void OSGPSOParticleOnResponseSurface::all(void (OSGPSOParticleOnResponseSurface::*func)())
{
    for (int i = 0; i < numpars; ++i)
    {
        (par[i]->*func)();
    }
}

void OSGPSOParticleOnResponseSurface::initRand(long int seed)
{
    //struct timeval acttime;
    //gettimeofday( &acttime, NULL );

    //mtrand = new MTRand(acttime.tv_sec + acttime.tv_usec);
    mtrand = new MTRand(seed);
}
