/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Tracer.h"
#include "Const.h"

Tracer::Tracer()
{
}

Tracer::~Tracer()
{
}

void Tracer::trace()
{
    init();

    TracerStep current;
    TracerStep first = result.front();
    double previousDistance(0.0);
    bool shrinking(false);

    int count(0);
    while (true)
    {
        ++count;
        step();

        // abort if we arrived at MAX_STEPS
        if (count > TRACE_MAX_STEPS)
            break;

        current = result.back();

        // abort if we leave the region
        if ((fabs(current.position[0]) > TRACE_MAX_REGION) || (fabs(current.position[1]) > TRACE_MAX_REGION) || (fabs(current.position[2]) > TRACE_MAX_REGION))
            break;

        // abort if we have a circle
        if ((config.electricField == 0.0) && (config.angle == 0.0))
        {
            double distance = (current.position - first.position).length2();
            if (distance < previousDistance)
            {
                shrinking = true;
            }
            else if (shrinking && (distance > previousDistance))
            {
                break;
            }
            previousDistance = distance;
        }
    }
}

void Tracer::init()
{
    result.clear();
    TracerStep init;

    init.position = TRACE_CENTER;

    osg::Matrix m;
    m.makeRotate(config.angle, BASE_VECTOR_PARTICLE_AXIS);
    init.velocity = m * (BASE_VECTOR_PARTICLE * config.velocity);

    result.push_back(init);
}

void Tracer::step()
{
    // calculate forces of old step
    calculateForces(result.back());

    // convenience variable
    TracerStep oldStep = result.back();

    // calculate derivative at old position
    osg::Vec3d dP = oldStep.velocity;
    osg::Vec3d dV = oldStep.combinedForce / config.mass;

    // calculate new/2 position
    TracerStep tmpStep = oldStep;
    tmpStep.position += dP * 0.5 * TRACE_DELTA;
    tmpStep.velocity += dV * 0.5 * TRACE_DELTA;

    // calculate force at new/2 position
    calculateForces(tmpStep);

    // calculate derivative at new/2 position
    dP = tmpStep.velocity;
    dV = tmpStep.combinedForce / config.mass;

    // calculate new position
    TracerStep newStep = oldStep;
    newStep.position += dP * TRACE_DELTA;
    newStep.velocity += dV * TRACE_DELTA;

    result.push_back(newStep);
}

void Tracer::calculateForces(TracerStep &step)
{
    // electric

    step.electricForce = BASE_VECTOR_ELECTRIC * config.electricField * config.charge;

    // magnetic

    osg::Vec3d projection = BASE_VECTOR_MAGNETIC * (BASE_VECTOR_MAGNETIC * step.velocity);
    osg::Vec3d orthogonalPart = step.velocity - projection;

    osg::Vec3d direction = orthogonalPart ^ (BASE_VECTOR_MAGNETIC * config.magneticField);
    direction.normalize();

    step.magneticForce = direction * (config.charge * orthogonalPart.length() * fabs(config.magneticField)); // abs because a negativ strength already influenced the direction

    // combined

    step.combinedForce = step.electricForce + step.magneticForce;
}
