/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__TRACER_H)
#define __TRACER_H

#include "Grid.h"
#include "Particle.h"

#ifdef _WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

class Tracer;

#include <api/coSimpleModule.h>
using namespace covise;

// for debugging !
#define G_ELEMMAX 100000
extern int g_numElem, g_numConn, g_numPoints;
extern int *g_elemList, *g_connList, *g_typeList;
extern float *g_xCoord, *g_yCoord, *g_zCoord, *g_dOut;

#define PLANE 1
#define VOLUME 2
#define CYLINDER 3

class Tracer : public coSimpleModule
{
private:
    // ports
    coInputPort *gridIn, *velIn;
    coOutputPort *outPort1, *outPort2, *outPort3;

    coOutputPort *outDebug, *outDataDebug;

    // parameters
    coBooleanParam *saveSearchFlag;
    coFloatParam *stepDuration;
    coIntScalarParam *numParticles;
    coIntScalarParam *numSteps, *startStep;
    coBooleanParam *loopDetection;
    coFloatVectorParam *startpoint1, *startpoint2;
    coFloatVectorParam *normal, *direction;
    coFloatVectorParam *cylAxis;
    coFloatVectorParam *cylAxisPoint;
    coFloatParam *cylRadius;
    coFloatParam *cylHeight;

    coChoiceParam *whatDataOut, *startStyle, *outputStyle;
    coChoiceParam *bBoxAlgorithm;
    coFloatParam *attackAngle;
    coFloatParam *streakAngle;

    // symmetric boundary conditions (sbc)
    coFloatVectorParam *sbcRotAxxis;
    coFloatParam *sbcRotAngle;
    coBooleanParam *sbcUndoFlag;

    // don't initialize the same data multiple times
    char *gridObjName, *dataObjName;

    // the grid we are working on
    Grid *grid;

    // and the particles
    Particle **particles;

    // this initializes all the particles (returns number of particles initialized)
    int startParticles();

    // build the output-objects
    void buildOutput(int nP, int numGridSteps);
    void buildOutput(int nP);

    virtual void param(const char *paramname, bool /*inMapLoading*/);

    // main-callback
    virtual int compute(const char *port);
    virtual void postInst();

public:
    Tracer(int argc, char *argv[]);
    virtual ~Tracer();
};

class WristWatch
{
private:
#ifdef _WIN32
    struct __timeb64 myClock;
#else
    timeval myClock;
#endif

public:
    WristWatch();
    ~WristWatch();

    void start();
    void stop(char *s);
};
#endif // __TRACER_H
