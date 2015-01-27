/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COMPUTETRACE_H
#define COMPUTETRACE_H

// includes
#include <api/coSimpleModule.h>
using namespace covise;
#include <util/coviseCompat.h>
#include <util/coRestraint.h>
#include <do/coDoSet.h>

class ComputeTrace : public coSimpleModule
{
public:
    ComputeTrace(int argc, char *argv[]);
    int compute(const char *port);

private:
    //ports
    coInputPort *p_pointsIn, *p_dataIn, *p_IDIn;
    coOutputPort *p_traceOut, *p_indexOut, *p_fadingOut, *p_dataOut;

    //parameters
    coIntSliderParam *p_start, *p_stop;
    coStringParam *p_particle;
    coBooleanParam *p_traceParticle;
    coBooleanParam *p_regardInterrupt;
    coBooleanParam *p_animate;
    coIntScalarParam *p_maxParticleNumber;
    coChoiceParam *p_animateViewer;
    coFloatVectorParam *p_animLookAt;
    coFloatVectorParam *p_boundingBoxDimensions;
    virtual void param(const char *name, bool inMapLoading);

    /*tracing particle*/
    coDoLines *computeCurrentTrace(const char *name, coDoLines *old_trace,
                                   int i, float x_new, float y_new, float z_new);
    coDistributedObject **computeDynamicTrace(const coDoSet *, bool compute);
    coDistributedObject **computeFloats(bool, float (*)(float, int));
    coDoLines *computeStaticTrace(const std::string &name, const coDoSet *spheres);
    coDistributedObject *extractData(const std::string &name, const coDoSet *set, bool animate);
    coDistributedObject *extractDataForParticle(const std::string &name, const coDoSet *set, int timestep);
    coDistributedObject *extractStaticData(const std::string &name, const coDoSet *set);
    static float returnIndex(float, int);
    static float fade(float, int);
    std::string createIndexedName(const char *, int, int);

    int getIndex(int t, int numIDs, int ID);

    int m_timestepStart, m_timestepStop, m_start, m_stop;
    int **IDs;
    vector<ssize_t> m_particleSelection;
    coRestraint m_particleIndices;

    bool m_firsttime;
};
#endif
