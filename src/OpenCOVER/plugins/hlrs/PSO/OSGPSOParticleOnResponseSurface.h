/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSGPSOParticleOnResponseSurface_h
#define OSGPSOParticleOnResponseSurface_h

#include <osg/Geode>
#include <osg/PositionAttitudeTransform>
#include "Particle.h"

class OSGPSOParticleOnResponseSurface : public osg::PositionAttitudeTransform, public pso::Particle
{
public:
    OSGPSOParticleOnResponseSurface();
    virtual ~OSGPSOParticleOnResponseSurface(){};

    void computePath();

    void move();

    void updateVelocity();

    void updatePosition();

    static double t;

    static void init(double (*setresponse)(double *), double *setlowerbound, double *setupperbound, bool *setinteger, long int seed);

    static void destroy();

    static void all(void (OSGPSOParticleOnResponseSurface::*func)());

protected:
    static osg::Geode *ParticleGeode;

    double *oldX;
    double *oldV;

    double **p;

    static std::vector<OSGPSOParticleOnResponseSurface *> par;

    virtual void enforceConstrains();

    static void initRand(long int seed);
};

#endif
