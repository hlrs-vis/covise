/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OSGPSOParticle.h"

#include <osg/ShapeDrawable>
#include <string.h>

std::vector<OSGPSOParticle *> OSGPSOParticle::par;
double OSGPSOParticle::t = 0;
osg::Geode *OSGPSOParticle::ParticleGeode = new osg::Geode();

OSGPSOParticle::OSGPSOParticle()
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

void OSGPSOParticle::computePath()
{
    for (int j = 0; j < nvar; ++j)
    {
        p[j][0] = (-2 * x[j] + 2 * oldX[j] + dt * (v[j] + oldV[j])) / (pow(dt, 3));
        p[j][1] = -(-3 * x[j] + 3 * oldX[j] + dt * (v[j] + 2 * oldV[j])) / (pow(dt, 2));
        p[j][2] = oldV[j];
        p[j][3] = oldX[j];
    }
}

void OSGPSOParticle::move()
{
    osg::Vec3 OSGPos(p[0][0] * pow(t, 3) + p[0][1] * pow(t, 2) + p[0][2] * t + p[0][3],
                     p[1][0] * pow(t, 3) + p[1][1] * pow(t, 2) + p[1][2] * t + p[1][3],
                     p[2][0] * pow(t, 3) + p[2][1] * pow(t, 2) + p[2][2] * t + p[2][3]);
    setPosition(OSGPos);

    osg::Quat OSGRot;
    OSGRot.makeRotate(osg::Vec3(0, 0, 1), osg::Vec3(3 * p[0][0] * pow(t, 2) + 2 * p[0][1] * t + p[0][2],
                                                    3 * p[1][0] * pow(t, 2) + 2 * p[1][1] * t + p[1][2],
                                                    3 * p[2][0] * pow(t, 2) + 2 * p[2][1] * t + p[2][2]));
    setAttitude(OSGRot);
}

void OSGPSOParticle::updateVelocity()
{
    memcpy(oldV, v, nvar * sizeof(double));
    pso::Particle::updateVelocity();
}

void OSGPSOParticle::updatePosition()
{
    memcpy(oldX, x, nvar * sizeof(double));
    pso::Particle::updatePosition();
}

void OSGPSOParticle::init(double (*setresponse)(double *), double *setlowerbound, double *setupperbound, bool *setinteger, long int seed)
{
    pso::Particle::init(setresponse, 3, setlowerbound, setupperbound, setinteger, 1);
    initRand(seed);

    par.clear();

    //ParticleGeode->addDrawable(new osg::ShapeDrawable(new osg::Sphere(osg::Vec3(0,0,0), 10)));
    ParticleGeode->addDrawable(new osg::ShapeDrawable(new osg::Cone(osg::Vec3(0, 0, 0), 0.01f, 0.05f)));
}

void OSGPSOParticle::destroy()
{
    pso::Particle::destroy();
}

void OSGPSOParticle::all(void (OSGPSOParticle::*func)())
{
    for (int i = 0; i < numpars; ++i)
    {
        (par[i]->*func)();
    }
}

void OSGPSOParticle::initRand(long int seed)
{
    //struct timeval acttime;
    //gettimeofday( &acttime, NULL );

    //mtrand = new MTRand(acttime.tv_sec + acttime.tv_usec);
    mtrand = new MTRand(seed);
}
