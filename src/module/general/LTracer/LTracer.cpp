/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "LTracer.h"

#include "Grid.h"
#include "Particle.h"

#define _PROFILE_ 1

#include <util/coviseCompat.h>

int g_numElem, g_numConn, g_numPoints;
int *g_elemList, *g_connList, *g_typeList;
float *g_xCoord, *g_yCoord, *g_zCoord, *g_dOut;

long int d_getVel, d_fnc, d_initGBI;

Tracer::Tracer(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Particle Tracer")
{
    // parameters
    saveSearchFlag = addBooleanParam("safeSearchFlag", "set this true to double-check particles that seem to have left the grid");
    saveSearchFlag->setValue(0);

    stepDuration = addFloatParam("stepDuration", "duration of 1 step in seconds");
    stepDuration->setValue(0.1f);

    numParticles = addInt32Param("numParticles", "number of particles to trace");
    numParticles->setValue(10);

    numSteps = addInt32Param("numSteps", "number of steps to trace (-1 for endless)");
    numSteps->setValue(100);

    loopDetection = addBooleanParam("loopDetection", "try to detect closed loops in particle-paths");
    loopDetection->setValue(0);

    startStep = addInt32Param("startStep", "step to start tracing in (transient grid only)");
    startStep->setValue(0);

    startpoint1 = addFloatVectorParam("startpoint1", "first starting-point");
    startpoint1->setValue(1.0, 1.0, 1.0);

    startpoint2 = addFloatVectorParam("startpoint2", "last starting-point");
    startpoint2->setValue(1.0, 2.0, 1.0);

    normal = addFloatVectorParam("normal", "normal for starting-plane");
    normal->setValue(0.0, 0.0, 1.0);

    direction = addFloatVectorParam("direction", "direction for starting-plane");
    direction->setValue(1.0, 0.0, 0.0);

    cylAxis = addFloatVectorParam("cyl_axis", "axis of starting cylinder");
    cylAxis->setValue(0.0, 0.0, 1.0);

    cylAxisPoint = addFloatVectorParam("cyl_bottompoint_on_axis", "point on starting cylinder");
    cylAxisPoint->setValue(1.0, 0.0, 0.0);

    cylRadius = addFloatParam("cyl_radius", "diameter of starting cylinder");
    cylRadius->setValue(1.0);

    cylHeight = addFloatParam("cyl_height", "height of starting cylinder");
    cylHeight->setValue(1.0);

    static const char *dDataOutVal[] = { "ID", "velocity", "magnitude" };
    whatDataOut = addChoiceParam("whatDataOut", "what data should be provided");
    whatDataOut->setValue(3, dDataOutVal, 0);

    static const char *dStartStyleVal[] = { "line", "plane", "volume", "cylinder" };
    startStyle = addChoiceParam("startStyle", "selects how the particles will start");
    startStyle->setValue(4, dStartStyleVal, 0);

    static const char *dOutputStyleVal[] = { "particles", "streamlines", "ribbons", "streaklines", "streakpoints" };
    outputStyle = addChoiceParam("outputStyle", "selects how the particles should be represented");
    outputStyle->setValue(5, dOutputStyleVal, 0);

    static const char *dbBoxAlgoVal[] = { "axisBox", "axisSphere", "miniBox", "miniSphere" };
    bBoxAlgorithm = addChoiceParam("bBoxAlgorithm", "selects the algorithm used to compute bounding-shapes");
    bBoxAlgorithm->setValue(4, dbBoxAlgoVal, 0);

    attackAngle = addFloatParam("attackAngle", "particles hitting boundaries in an angle lower than this value will not leave the grid (0.0=disabled)");
    attackAngle->setValue(20.0);

    streakAngle = addFloatParam("streakAngle", "if that angle is fallen short of in a streakline, the line is being separated (0.0=disabled)");
    streakAngle->setValue(160.0);

    // symmetric boundary conditions
    sbcRotAxxis = addFloatVectorParam("rotAxxis", "axxis for the rotation");
    sbcRotAxxis->setValue(0.0, 0.0, 1.0);
    sbcRotAngle = addFloatParam("rotAngle", "angle of rotation");
    sbcRotAngle->setValue(0.0);
    sbcUndoFlag = addBooleanParam("sbcUndoFlag", "set this true if you want to see the real particlepath and not the reentrant one");
    sbcUndoFlag->setValue(0);

    // ports
    gridIn = addInputPort("gridIn", "UnstructuredGrid|RectilinearGrid|StructuredGrid", "grid input");
    velIn = addInputPort("velIn", "Vec3|Vec3", "velocity data input");
    gridObjName = NULL;
    dataObjName = NULL;

    outPort1 = addOutputPort("output1", "Points|Lines", "usually these are the points");
    outPort2 = addOutputPort("output2", "Points|Lines|Polygons|TriangleStrips", "these are the lines/ribbons");
    outPort3 = addOutputPort("output3", "Vec3|Float", "data output");
    outDebug = addOutputPort("debugOut", "UnstructuredGrid", "debugging output");
    outDataDebug = addOutputPort("debugDataOut", "Float", "debugging data output");

    // we just do it
    setComputeTimesteps(1);
    setComputeMultiblock(1);
    setCopyAttributes(0);

    // done
    return;
}

void
Tracer::postInst()
{
    // default startstyle is line, so hide cylinder params
    startpoint1->show();
    startpoint1->enable();
    startpoint2->show();
    startpoint2->enable();
    normal->show();
    normal->enable();
    direction->show();
    direction->enable();

    cylAxis->hide();
    cylAxis->disable();
    cylAxisPoint->hide();
    cylAxisPoint->disable();
    cylRadius->hide();
    cylRadius->disable();
    cylHeight->hide();
    cylHeight->disable();
}

Tracer::~Tracer()
{
    // clean up
    if (gridObjName && dataObjName)
    {
        delete grid;
        delete gridObjName;
        delete dataObjName;
    }
}

void
Tracer::param(const char *paramname, bool)
{
    if (strcmp(paramname, startStyle->getName()) == 0)
    {
        switch (startStyle->getValue())
        {
        case CYLINDER:
            startpoint1->hide();
            startpoint1->disable();
            startpoint2->hide();
            startpoint2->disable();
            normal->hide();
            normal->disable();
            direction->hide();
            direction->disable();

            cylAxis->show();
            cylAxis->enable();
            cylAxisPoint->show();
            cylAxisPoint->enable();
            cylRadius->show();
            cylRadius->enable();
            cylHeight->show();
            cylHeight->enable();
            break;

        default:
            startpoint1->show();
            startpoint1->enable();
            startpoint2->show();
            startpoint2->enable();
            normal->show();
            normal->enable();
            direction->show();
            direction->enable();

            cylAxis->hide();
            cylAxis->disable();
            cylAxisPoint->hide();
            cylAxisPoint->disable();
            cylRadius->hide();
            cylRadius->disable();
            cylHeight->hide();
            cylHeight->disable();
            break;
        }
    }
}

int Tracer::compute(const char *)
{
    const coDistributedObject *gObjIn, *vObjIn;
    float dt;
    int nP, nS, ld;
    int needGradient, needRot;

#if defined(_PROFILE_)
    WristWatch _ww, _wwAll;
    _wwAll.start();
#endif

    // for debugging
    g_numElem = 0;
    g_numConn = 0;
    g_numPoints = 0;
    g_elemList = new int[G_ELEMMAX];
    g_typeList = new int[G_ELEMMAX];
    g_connList = new int[G_ELEMMAX * 8];
    g_xCoord = new float[G_ELEMMAX * 8];
    g_yCoord = new float[G_ELEMMAX * 8];
    g_zCoord = new float[G_ELEMMAX * 8];
    g_dOut = new float[G_ELEMMAX];

    d_getVel = 0L;
    d_fnc = 0L;
    d_initGBI = 0L;

    int s, p;

    // get Objects
    gObjIn = gridIn->getCurrentObject();
    vObjIn = velIn->getCurrentObject();
    if (!gObjIn || !vObjIn)
    {
        // error
        sendError("not all required input-objects available");
        return (coModule::FAIL);
    }

    // see what we should compute...
    nS = numSteps->getValue();
    ld = loopDetection->getValue();
    needGradient = 0;
    needRot = (outputStyle->getValue() == 2); // rot only needed for ribbons

    cerr << "#" << endl << "### Tracer: Initializing..." << endl << "#" << endl;

    // init the grid
    s = 1;

    if (gridObjName && dataObjName)
    {
#if 0
      if( !strcmp(gridObjName, gObjIn->getName()) && !strcmp(dataObjName, vObjIn->getName()) )
         s = 0;
      else
#endif
        {
            delete[] gridObjName;
            delete[] dataObjName;
            delete grid;
        }
    }

#if defined(_PROFILE_)
    Covise::sendInfo("===---------------------------------------------");
    _ww.start();
#endif
    if (s)
    {
        // init it
        gridObjName = new char[strlen(gObjIn->getName()) + 1];
        strcpy(gridObjName, gObjIn->getName());
        dataObjName = new char[strlen(vObjIn->getName()) + 1];
        strcpy(dataObjName, vObjIn->getName());
        grid = new Grid(gObjIn, vObjIn, saveSearchFlag->getValue(), bBoxAlgorithm->getValue(), (float)(((90 - attackAngle->getValue()) / 180.0) * M_PI), needGradient, needRot);
    }
    else
        cerr << "#" << endl << "### Tracer: Initialization skipped due to unchanged input" << endl << "#" << endl;

#if defined(_PROFILE_)
    _ww.stop((char *)"initialization");
#endif

    // tell the grid to use SBC if any conditions are given
    float sbcAx[3], sbcAng;
    for (s = 0; s < 3; s++)
        sbcAx[s] = sbcRotAxxis->getValue(s);
    sbcAng = sbcRotAngle->getValue();
    if (sbcAng != 0.0 && (sbcAx[0] != 0.0 || sbcAx[1] != 0.0 || sbcAx[2] != 0.0))
        grid->useSBC(sbcAx, sbcAng);

    dt = stepDuration->getValue();

    cerr << "#" << endl << "### Tracer: starting particles..." << endl << "#" << endl;

// start the particles
#if defined(_PROFILE_)
    _ww.start();
#endif
    nP = startParticles();
#if defined(_PROFILE_)
    _ww.stop((char *)"startParticles");
#endif

    cerr << "#" << endl << "### Tracer: performing traces" << endl << "#" << endl;
    cerr << "# nS=" << nS << "   , nP=" << nP << endl << "#" << endl;
// perform the traces
#if defined(_PROFILE_)
    _ww.start();
#endif

    // only count # of calls during tracing
    d_initGBI = 0L;

    for (s = 0; s < nS; s++) // loop over the steps
    {
        for (p = 0; p < nP; p++) // loop over all particles
        {
            if (particles[p])
            {
                if (!particles[p]->hasStopped())
                {
                    particles[p]->trace(dt);
                    if (ld && !particles[p]->hasStopped())
                        particles[p]->loopDetection();
                }
            }
            //else
            //   cerr << "# particles[" << p << "] stopped" << endl;
        }
    }
#if defined(_PROFILE_)
    _ww.stop((char *)"traces");
#endif

    cerr << "#" << endl << "### Tracer: building output" << endl << "#" << endl;

// build the output
#if defined(_PROFILE_)
    _ww.start();
#endif
    buildOutput(nP, grid->getNumSteps());
#if defined(_PROFILE_)
    _ww.stop((char *)"output generation");
#endif

    // build debugging output and clean up
    if (g_numElem)
    {
        coDoUnstructuredGrid *dbgGrd = new coDoUnstructuredGrid(outDebug->getObjName(), g_numElem,
                                                                g_numConn, g_numPoints, g_elemList, g_connList, g_xCoord, g_yCoord, g_zCoord, g_typeList);
        outDebug->setCurrentObject(dbgGrd);

        coDoFloat *dbgDta = new coDoFloat(outDataDebug->getObjName(), g_numElem, g_dOut);
        outDataDebug->setCurrentObject(dbgDta);
    }
    delete[] g_xCoord;
    delete[] g_yCoord;
    delete[] g_zCoord;
    delete[] g_connList;
    delete[] g_elemList;
    delete[] g_typeList;
    delete[] g_dOut;
    /*
      cerr << "#" << endl << "### Tracer: end" << endl << "#" << endl;

      cerr << "calls to ...::getVelocity : " << d_getVel << "   avg: " << ((d_getVel-nP)/(nS*nP)) << endl;
      cerr << "calls to ...::findNextCell: " << d_fnc << "   avg: " << (d_fnc/(nS*nP)) << endl;
      cerr << "calls to ...::initGBI: " << d_initGBI << "   avg: " << (d_initGBI/(nS*nP)) << endl;
   */

    // clean up
    for (p = 0; p < nP; p++)
        if (particles[p])
            delete particles[p];
    delete[] particles;
//delete grid;

#if defined(_PROFILE_)
    _wwAll.stop((char *)"complete run");
#endif

    // done
    return (coModule::SUCCESS);
}

int Tracer::startParticles()
{
    int nP = 0;

    cerr << "startStyle: " << startStyle->getValue() << endl;
    cerr << "outputStyle: " << outputStyle->getValue() << endl;

    // add other startStyles here
    switch (startStyle->getValue())
    {
    case PLANE:
    {

        float startDirection[3], startNormal[3], startp1[3], startp2[3];
        float r, s0, s1;
        int n0, n1;
        float v0[3], v1[3], myDir[3], o[3];

        int i, j, n, numStart;
        Particle *t;
        int sS;

        sS = startStep->getValue();

        for (i = 0; i < 3; i++)
        {
            startDirection[i] = direction->getValue(i);
            startNormal[i] = normal->getValue(i);
            startp1[i] = startpoint1->getValue(i);
            startp2[i] = startpoint2->getValue(i);
            o[i] = startp1[i];
        }

        numStart = numParticles->getValue();

        cerr << "numStart: " << numStart << endl;

        // given:  startpoint1, startpoint2, startDirection, startNormal
        //
        // we have to compute: s0, s1 first, then n0, n1
        //

        // normize all vectors first
        r = 1.0f / sqrt((startDirection[0] * startDirection[0]) + (startDirection[1] * startDirection[1]) + (startDirection[2] * startDirection[2]));
        startDirection[0] *= r;
        startDirection[1] *= r;
        startDirection[2] *= r;

        r = 1.0f / sqrt((startNormal[0] * startNormal[0]) + (startNormal[1] * startNormal[1]) + (startNormal[2] * startNormal[2]));
        startNormal[0] *= r;
        startNormal[1] *= r;
        startNormal[2] *= r;

        myDir[0] = startNormal[1] * startDirection[2] - startNormal[2] * startDirection[1];
        myDir[1] = startNormal[2] * startDirection[0] - startNormal[0] * startDirection[2];
        myDir[2] = startNormal[0] * startDirection[1] - startNormal[1] * startDirection[0];
        // that should be normized by definition, so we have nothing to do

        // the following algorithm is standard, though it is taken
        // from GraphicGems I, side 304
        //

        // v1 x v2 = v0  (for speed up)
        v0[0] = startDirection[1] * myDir[2] - startDirection[2] * myDir[1];
        v0[1] = startDirection[2] * myDir[0] - startDirection[0] * myDir[2];
        v0[2] = startDirection[0] * myDir[1] - startDirection[1] * myDir[0];

        // v1 = p2 - p1
        v1[0] = startp2[0] - startp1[0];
        v1[1] = startp2[1] - startp1[1];
        v1[2] = startp2[2] - startp1[2];

        // compute r = det(v1, myDir, v0)
        r = (v1[0] * myDir[1] * v0[2]) + (myDir[0] * v0[1] * v1[2]) + (v0[0] * v1[1] * myDir[2]) - (v0[0] * myDir[1] * v1[2]) - (v1[0] * v0[1] * myDir[2]) - (myDir[0] * v1[1] * v0[2]);

        // and divide by v0^2
        s0 = (v0[0] * v0[0]) + (v0[1] * v0[1]) + (v0[2] * v0[2]);
        if (s0 == 0.0)
        {
            fprintf(stderr, "s0 = 0.0 !!!\n");
            s0 = 0.01f;
        }

        r /= s0;

        // so upper left point is at
        s1 = -r; // ulp = startpoint1 + s1*myDir

        // and compute r = det(v1, startDirection, v0)
        r = (v1[0] * startDirection[1] * v0[2]) + (startDirection[0] * v0[1] * v1[2]) + (v0[0] * v1[1] * startDirection[2]) - (v0[0] * startDirection[1] * v1[2]) - (v1[0] * v0[1] * startDirection[2]) - (startDirection[0] * v1[1] * v0[2]);
        r /= s0;

        // so lower right point is at
        s0 = r; // lrp = startpoint1 + s0*startDirection

        // we flipped 'em somehow....damn
        r = s0;
        s0 = -s1;
        s1 = -r;

        // now compute "stepsize" and "stepcount"
        n1 = (int)sqrt(fabsf(((float)numStart) * (s1 / s0)));
        if (n1 == 0)
            n1 = 1;
        n0 = numStart / n1;

        s0 /= ((float)n0);
        s1 /= ((float)n1);

        v0[0] = startDirection[0];
        v0[1] = startDirection[1];
        v0[2] = startDirection[2];

        v1[0] = myDir[0];
        v1[1] = myDir[1];
        v1[2] = myDir[2];

        numStart = n1 * n0;

        nP = numStart;

        particles = new Particle *[nP];

        cerr << "trying to start: " << nP << endl;

        n = 0;
        for (j = 0; j < n1; j++)
        {
            for (i = 0; i < n0; i++)
            {
                t = grid->startParticle(sS, o[0] + v0[0] * s0 * (float)i + v1[0] * s1 * (float)j, o[1] + v0[1] * s0 * (float)i + v1[1] * s1 * (float)j, o[2] + v0[2] * s0 * (float)i + v1[2] * s1 * (float)j);
                if (t)
                {
                    particles[n] = t;
                    n++;
                }
            }
        }
        nP = n;
    }
    break;

    case VOLUME:
    {
        float v[3], o[3], temp;
        Particle *t;

        int n0, n1, n2;
        float s0, s1, s2;
        float V;

        int i, j, k, n;

        int sS;

        nP = numParticles->getValue();
        sS = startStep->getValue();

        if (nP == 1)
        {

            particles = new Particle *[1];
            t = grid->startParticle(sS, startpoint1->getValue(0), startpoint1->getValue(1), startpoint1->getValue(2));
            if (t)
                particles[0] = t;
        }
        else
        {
            for (i = 0; i < 3; i++)
            {
                o[i] = startpoint1->getValue(i);
                v[i] = startpoint2->getValue(i);
            }

            // Volume size
            V = fabs((v[0] - o[0]) * (v[1] - o[1]) * (v[2] - o[2]));

            temp = pow((float)(nP / V), 1.f / 3.f);
            n0 = (int)fabs(temp * (v[0] - o[0]));
            n1 = (int)fabs(temp * (v[1] - o[1]));
            n2 = (int)fabs(temp * (v[2] - o[2]));

            if (n0 == 0)
                n0 = 1;
            if (n1 == 0)
                n1 = 1;
            if (n2 == 0)
                n2 = 1;

            cerr << "n0: " << n0 << "  n1: " << n1 << "  n2: " << n2 << endl;

            s0 = (v[0] - o[0]) / (n0 - 1);
            s1 = (v[1] - o[1]) / (n1 - 1);
            s2 = (v[2] - o[2]) / (n2 - 1);

            cerr << "s0: " << s0 << "  s1: " << s1 << "  s2: " << s2 << endl;

            nP = n0 * n1 * n2;

            particles = new Particle *[nP];

            cerr << "trying to start: " << nP << endl;

            n = 0;
            for (k = 0; k < n0; k++)
            {
                for (j = 0; j < n1; j++)
                {
                    for (i = 0; i < n2; i++)
                    {
                        t = grid->startParticle(sS,
                                                o[0] + s0 * (float)k,
                                                o[1] + s1 * (float)j,
                                                o[2] + s2 * (float)i);
                        if (t)
                        {
                            particles[n] = t;
                            n++;
                        }
                    }
                }
            }
            nP = n;
        }
    }
    break;

    case CYLINDER:
    {

        float cylinderAxis[3], cylinderAxisPoint[3];
        float cylinderHeight, cylinderRadius;

        float r, s0, s1;
        float s;
        int n0, n1, no_p;
        //float v0[3], v1[3], myDir[3], o[3];

        int i, j, n, numStart;
        Particle *t;
        int sS;

        sS = startStep->getValue();

        for (i = 0; i < 3; i++)
        {
            cylinderAxis[i] = cylAxis->getValue(i);
            cylinderAxisPoint[i] = cylAxisPoint->getValue(i);
        }
        cylinderHeight = cylHeight->getValue();
        cylinderRadius = cylRadius->getValue();

        numStart = numParticles->getValue();
        no_p = numParticles->getValue();

        cerr << "numStart: " << numStart << endl;

        // normize all vectors first
        r = 1.0f / sqrt((cylinderAxis[0] * cylinderAxis[0]) + (cylinderAxis[1] * cylinderAxis[1]) + (cylinderAxis[2] * cylinderAxis[2]));
        cylinderAxis[0] *= r;
        cylinderAxis[1] *= r;
        cylinderAxis[2] *= r;

        // lengths
        s = 2 * M_PI * cylinderRadius;

        // and now number of points in axis direction and around cylinder
        n1 = int(sqrt(no_p * s / cylinderHeight)) + 1;
        if (n1 <= 1)
            n1 = 2;
        n0 = no_p / n1;
        if (n0 <= 1)
            n0 = 2;

        no_p = n0 * n1;
        s0 = 1.0f / (n0 - 1);
        s1 = 1.0f / (n1 - 1);

        // get plane of cylinder bottom
        // x*nx+y*ny+z*nz-d=0
        float d = cylinderAxisPoint[0] * cylinderAxis[0] + cylinderAxisPoint[1] * cylinderAxis[1] + cylinderAxisPoint[2] * cylinderAxis[2];

        // we need two vectors at right angle to axis and to each other ...
        float p[3], vec1[3], vec2[3];
        vec1[0] = 0.0;
        vec1[1] = 0.0;
        vec1[2] = 0.0;
        vec2[0] = 0.0;
        vec2[1] = 0.0;
        vec2[2] = 0.0;
        float TOL = 0.0001f;
        // at first, determine one point p in plane
        int xsmall = 0;
        int ysmall = 0;
        int zsmall = 0;
        int nsmall = 0;
        if ((fabs(cylinderAxis[0]) < TOL))
        {
            xsmall = 1;
            nsmall++;
        }
        if ((fabs(cylinderAxis[1]) < TOL))
        {
            ysmall = 1;
            nsmall++;
        }
        if ((fabs(cylinderAxis[2]) < TOL))
        {
            zsmall = 1;
            nsmall++;
        }
        if (nsmall == 0)
        {
            p[0] = -cylinderAxis[0]; // this is arbitrary
            p[1] = cylinderAxis[1];
            p[2] = (d - cylinderAxis[0] * vec1[0] - cylinderAxis[1] * vec1[1]) / cylinderAxis[2];
        }
        else if (nsmall == 1)
        {
            if (xsmall)
            {
                p[0] = cylinderAxis[0];
                p[1] = -cylinderAxis[1];
                p[2] = (d - cylinderAxis[0] * vec1[0] - cylinderAxis[1] * vec1[1]) / cylinderAxis[2];
            }
            else if (ysmall)
            {
                p[0] = -cylinderAxis[0];
                p[1] = cylinderAxis[1];
                p[2] = (d - cylinderAxis[0] * vec1[0] - cylinderAxis[1] * vec1[1]) / cylinderAxis[2];
            }
            else // zsmall
            {
                p[0] = -cylinderAxis[0];
                p[1] = (d - cylinderAxis[0] * vec1[0] - cylinderAxis[2] * vec1[2]) / cylinderAxis[1];
                p[2] = cylinderAxis[2];
            }
        }
        else if (nsmall == 2)
        {
            if (!xsmall)
            {
                vec1[0] = 0.;
                vec1[1] = 1.;
                vec1[2] = 0.;

                vec2[0] = 0.;
                vec2[1] = 0.;
                vec2[2] = 1.;
            }
            else if (!ysmall)
            {
                vec1[0] = 1.;
                vec1[1] = 0.;
                vec1[2] = 0.;

                vec2[0] = 0.;
                vec2[1] = 0.;
                vec2[2] = 1.;
            }
            else // !zsmall
            {
                vec1[0] = 1.;
                vec1[1] = 0.;
                vec1[2] = 0.;

                vec2[0] = 0.;
                vec2[1] = 1.;
                vec2[2] = 0.;
            }
        }
        else // nsmall==3
        {
            sendError("Your cylinder axis seems to be zero or close to zero. Please check.");
            return FAIL;
        }

        if (nsmall != 2)
        {
            // 1st vector in bottom circle is (p-axispoint)
            vec1[0] = p[0] - cylinderAxisPoint[0];
            vec1[1] = p[1] - cylinderAxisPoint[1];
            vec1[2] = p[2] - cylinderAxisPoint[2];

            // 2nd vector is perpendicular to vec1 and to axis
            vec2[0] = cylinderAxis[1] * vec1[2] - cylinderAxis[2] * vec1[1];
            vec2[1] = cylinderAxis[2] * vec1[0] - cylinderAxis[0] * vec1[2];
            vec2[2] = cylinderAxis[0] * vec1[1] - cylinderAxis[1] * vec1[0];

            // normalize vectors
            r = 1.0f / sqrt((vec1[0] * vec1[0]) + (vec1[1] * vec1[1]) + (vec1[2] * vec1[2]));
            vec1[0] *= r;
            vec1[1] *= r;
            vec1[2] *= r;
            r = 1.0f / sqrt((vec2[0] * vec2[0]) + (vec2[1] * vec2[1]) + (vec2[2] * vec2[2]));
            vec2[0] *= r;
            vec2[1] *= r;
            vec2[2] *= r;
        }

        // check vectors for rectangularity
        if ((vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]) > TOL)
        {
            sendError("error in fillCylinder. Some vecotrs don't seem to be orthogonal.");
            return FAIL;
        }
        if ((vec1[0] * cylinderAxis[0] + vec1[1] * cylinderAxis[1] + vec1[2] * cylinderAxis[2]) > TOL)
        {
            sendError("error in fillCylinder. Some vecotrs don't seem to be orthogonal.");
            return FAIL;
        }
        if ((vec2[0] * cylinderAxis[0] + vec2[1] * cylinderAxis[1] + vec2[2] * cylinderAxis[2]) > TOL)
        {
            sendError("error in fillCylinder. Some vecotrs don't seem to be orthogonal.");
            return FAIL;
        }

        // now fill the grids
        float *x_ini = new float[no_p];
        float *y_ini = new float[no_p];
        float *z_ini = new float[no_p];

        s0 *= cylinderHeight;
        float angle = 0.;
        for (i = 0; i < n0; ++i)
        {
            for (j = 0; j < n1; ++j)
            {
                angle = s1 * j * 2 * M_PI;
                x_ini[i * n1 + j] = cylinderAxisPoint[0] + cylinderRadius * vec1[0] * cos(angle) + cylinderRadius * vec2[0] * sin(angle) + cylinderAxis[0] * i * s0;
                y_ini[i * n1 + j] = cylinderAxisPoint[1] + cylinderRadius * vec1[1] * cos(angle) + cylinderRadius * vec2[1] * sin(angle) + cylinderAxis[1] * i * s0;
                z_ini[i * n1 + j] = cylinderAxisPoint[2] + cylinderRadius * vec1[2] * cos(angle) + cylinderRadius * vec2[2] * sin(angle) + cylinderAxis[2] * i * s0;
            }
        }

        numStart = no_p;

        nP = no_p;

        particles = new Particle *[nP];

        cerr << "trying to start: " << nP << endl;

        n = 0;
        for (i = 0; i < no_p; i++)
        {
            t = grid->startParticle(sS, x_ini[i],
                                    y_ini[i],
                                    z_ini[i]);
            if (t)
            {
                particles[n] = t;
                n++;
            }
        }
        nP = no_p;
    }
    break;

    default: // unknown / line
    {
        float v[3], o[3], s;
        Particle *t;

        int i, sS, n;

        nP = numParticles->getValue();
        sS = startStep->getValue();

        if (nP == 1)
        {

            particles = new Particle *[1];
            t = grid->startParticle(sS, startpoint1->getValue(0), startpoint1->getValue(1), startpoint1->getValue(2));
            if (t)
                particles[0] = t;
        }
        else
        {

            for (i = 0; i < 3; i++)
            {
                o[i] = startpoint1->getValue(i);
                v[i] = startpoint2->getValue(i);
            }
            s = 1.0f / (float)(nP - 1);

            v[0] -= o[0];
            v[1] -= o[1];
            v[2] -= o[2];

            particles = new Particle *[nP];
            for (i = 0, n = 0; i < nP; i++)
            {
                t = grid->startParticle(sS, o[0] + v[0] * s * (float)i, o[1] + v[1] * s * (float)i, o[2] + v[2] * s * (float)i);
                if (t)
                {
                    //                  cerr << "sP: " << i << "   pos: " << o[0]+v[0]*s*(float)i << " " << o[1]+v[1]*s*(float)i << " " << o[2]+v[2]*s*(float)i << endl;
                    particles[n] = t;
                    n++;
                }
            }
            nP = n;
        }
    }
    break;
    }

    if (outputStyle->getValue() == 3 || outputStyle->getValue() == 4)
    {
        cerr << "initializing streaklines" << endl;

        // stream-lines
        //  take all particles and start them again in all timesteps
        Particle **new_particles;
        Particle *t;
        float x, y, z, v;
        int i, n, s;
        int sS;

        sS = startStep->getValue();

        new_particles = new Particle *[nP * grid->getNumSteps()];
        for (i = 0; i < nP; i++)
            new_particles[i] = particles[i];
        delete[] particles;
        particles = new_particles;

        // loop over all timesteps
        for (n = 1; n < grid->getNumSteps(); n++)
        {
            // loop over all particles
            for (i = 0; i < nP; i++)
            {
                particles[i]->getWayPoint(0, &s, &x, &y, &z, &v, &v, &v);
                t = grid->startParticle((sS + n) % grid->getNumSteps(), x, y, z);
                if (t)
                    particles[nP * n + i] = t;
                else
                    particles[nP * n + i] = NULL;
            }
        }

        nP *= grid->getNumSteps();
        cerr << nP << " particles" << endl;
    }

    /*
   for( int i=0; i<nP; i++ )
   {
      cerr << "particle # " << i << ": (x): " << particles[i]->gbiCur->x << " " << particles[i]->gbiCur->y << " ";
      cerr << particles[i]->gbiCur->z << "   block: " << particles[i]->gbiCur->blockNo << " / " << particles[i]->gbiCur->block->blockNo();
      cerr << "  step: " << particles[i]->gbiCur->stepNo << "/" << particles[i]->gbiCur->step->stepNo() << endl;
   }
   */

    // done
    return (nP);
}

void Tracer::buildOutput(int nP, int numGridSteps)
{
    int i, n;

    n = 0;
    for (i = 0; i < nP; i++)
    {
        if (particles[i])
            if (particles[i]->getNumSteps() > n)
                n = particles[i]->getNumSteps();
    }

    cerr << "### Tracer::buildOutput,   " << nP << " particles,  " << n << " steps"
         << "     " << numGridSteps << " Timesteps" << endl;

    //   static const char *dOutputStyleVal[] = { "points", "lines", "growing lines", "growing lines with points", "points with tails", "ribbons" };
    switch (outputStyle->getValue())
    {
    //=========================================================================================================
    case 1: // lines
    {
        char bfr[512];
        coDoLines *lines;
        coDoFloat *dat;
        int numLines, numCoord, numVert;
        int *vl, *ll;
        int m, s;
        float *x, *y, *z, *d;
        float u, v, w;

        int undo = sbcUndoFlag->getValue();

        numLines = 0;
        numCoord = 0;
        numVert = 0;
        for (i = 0; i < nP; i++)
        {
            if (particles[i]->getNumSteps() > 1)
            {
                numLines++;
                numCoord += particles[i]->getNumSteps();
                numVert += particles[i]->getNumSteps();
            }
        }

        lines = new coDoLines(outPort1->getObjName(), numCoord, numVert, numLines);
        dat = new coDoFloat(outPort3->getObjName(), numVert);
        lines->getAddresses(&x, &y, &z, &vl, &ll);
        lines->addAttribute("COLOR", "blue");
        dat->getAddress(&d);

        numLines = 0;
        numCoord = 0;
        numVert = 0;
        for (i = 0; i < nP; i++)
        {
            if (particles[i]->getNumSteps() > 1)
            {
                ll[numLines] = numVert;
                numLines++;

                for (m = 0; m < particles[i]->getNumSteps(); m++)
                {
                    vl[numVert] = numCoord;
                    numVert++;
                    particles[i]->getWayPoint(m, &s, x + numCoord, y + numCoord, z + numCoord, &u, &v, &w, NULL, NULL, NULL);
                    if (s < 0)
                    {
                        while (s < 0 && undo)
                        {
                            grid->sbcApply(x + numCoord, y + numCoord, z + numCoord);
                            s++;
                        }
                    }
                    else
                    {
                        while (s > 0 && undo)
                        {
                            grid->sbcInvApply(x + numCoord, y + numCoord, z + numCoord);
                            s--;
                        }
                    }
                    switch (whatDataOut->getValue())
                    {
                    case 0: // ID
                        d[numCoord] = (float)i;
                        break;

                    /*
							case 1:								// velocity

							break;
							*/

                    case 2: // magnitude
                        d[numCoord] = sqrt(u * u + v * v + w * w);
                        break;

                    default:
                        d[numCoord] = (float)i; // default is ID
                    }

                    numCoord++;
                }
            }
        }

        switch (startStyle->getValue())
        {
        case PLANE:
            sprintf(bfr, "P%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            break;
        default:
            sprintf(bfr, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
        }
        lines->addAttribute("FEEDBACK", bfr);

        outPort1->setCurrentObject(lines);
        outPort3->setCurrentObject(dat);
    }
    break;
    //=========================================================================================================
    case 3: // streaklines
    {
        if (numGridSteps == 1)
        {
            buildOutput(nP);
            return;
        }

        char bfr[512];
        int undo = sbcUndoFlag->getValue();
        int numPart = nP / numGridSteps;

        int k, o, s, cycles, f, _k;

        float sepAng = (float)cos((streakAngle->getValue() * M_PI) / 180.0);

        coDistributedObject *rObj = NULL;
        coDistributedObject **sObj = NULL;
        coDistributedObject **dObj = NULL;
        coDoFloat *dat;
        coDoLines *lines;
        int numLines, numVert;
        //int *vl, *ll;
        float *x, *y, *z, *d, t;

        sObj = new coDistributedObject *[numGridSteps + 1];
        dObj = new coDistributedObject *[numGridSteps + 1];
        sObj[numGridSteps] = NULL;
        dObj[numGridSteps] = NULL;

        int *numPathPart, *offsPathPart, *idxPathPart;
        numPathPart = new int[numPart];
        offsPathPart = new int[numPart];
        idxPathPart = new int[numPart];

        for (i = 0; i < numGridSteps; i++)
        {
            for (k = 0; k < numPart; k++)
            {
                numPathPart[k] = 0;
                offsPathPart[k] = 0;
            }

            o = 0;

            for (f = i; f >= 0; f--)
            {
                for (k = 0; k < numPart; k++)
                {
                    if (particles[k + f * numPart])
                    {
                        if (particles[k + f * numPart]->getNumSteps() > i - f)
                        {
                            o++;
                            numPathPart[k]++;
                        }
                    }
                }
            }
            n = 1;
            for (cycles = 0; n; cycles++)
            {
                n = 0;
                for (k = 0; k < numPart; k++)
                {
                    if (particles[k])
                    {
                        if (particles[k]->getNumSteps() > (numGridSteps * cycles) + i)
                        {
                            o++;
                            numPathPart[k]++;
                            n = 1;
                        }
                    }
                }
                for (k = 0; k < numPart; k++)
                {
                    for (f = numGridSteps - 1; f > 0; f--)
                    {
                        if (particles[k + f * numPart])
                        {
                            if (particles[k + f * numPart]->getNumSteps() > (numGridSteps * cycles) + i + (numGridSteps - f))
                            {
                                o++;
                                numPathPart[k]++;
                                n = 1;
                            }
                        }
                    }
                }
            }

            if (o > 0)
            {
                cerr << " found " << o << " points in step " << i << endl;

                // prepare on-the-fly sorting of the points
                s = 0;
                for (k = 0; k < numPart; k++)
                {
                    offsPathPart[k] = s;
                    s += numPathPart[k];
                    idxPathPart[k] = 0;
                }

                x = new float[o];
                y = new float[o];
                z = new float[o];
                d = new float[o];

                _k = o;
                o = 0;

                // (beginning)
                for (f = i; f >= 0; f--)
                {
                    for (k = 0; k < numPart; k++)
                    {
                        if (particles[k + f * numPart])
                        {
                            if (particles[k + f * numPart]->getNumSteps() > i - f)
                            {
                                o = offsPathPart[k] + idxPathPart[k];
                                idxPathPart[k]++;

                                particles[k + f * numPart]->getWayPoint(i - f, &s, x + o, y + o, z + o, &t, &t, &t, NULL, NULL, NULL);
                                if (s < 0)
                                {
                                    while (s < 0 && undo)
                                    {
                                        grid->sbcApply(x + o, y + o, z + o);
                                        s++;
                                    }
                                }
                                else
                                {
                                    while (s > 0 && undo)
                                    {
                                        grid->sbcInvApply(x + o, y + o, z + o);
                                        s--;
                                    }
                                }
                                d[o] = (float)k;
                            }
                        }
                    }
                }

                // point, "others", point-cycled, "others"-cycled...etc etc
                n = 1;
                for (cycles = 0; n; cycles++)
                {
                    n = 0;
                    for (k = 0; k < numPart; k++)
                    {
                        if (particles[k])
                        {
                            if (particles[k]->getNumSteps() > (numGridSteps * cycles) + i)
                            {
                                // at least once
                                o = offsPathPart[k] + idxPathPart[k];
                                idxPathPart[k]++;

                                particles[k]->getWayPoint(i + (cycles * numGridSteps), &s, x + o, y + o, z + o, &t, &t, &t, NULL, NULL, NULL);
                                if (s < 0)
                                {
                                    while (s < 0 && undo)
                                    {
                                        grid->sbcApply(x + o, y + o, z + o);
                                        s++;
                                    }
                                }
                                else
                                {
                                    while (s > 0 && undo)
                                    {
                                        grid->sbcInvApply(x + o, y + o, z + o);
                                        s--;
                                    }
                                }

                                d[o] = (float)k;
                                n = 1;
                            }
                        }
                    }

                    for (k = 0; k < numPart; k++)
                    {
                        for (f = numGridSteps - 1; f > 0; f--)
                        {
                            if (particles[k + f * numPart])
                            {
                                if (particles[k + f * numPart]->getNumSteps() > (numGridSteps * cycles) + i + (numGridSteps - f))
                                {
                                    o = offsPathPart[k] + idxPathPart[k];
                                    idxPathPart[k]++;

                                    particles[k + f * numPart]->getWayPoint((numGridSteps * cycles) + i + (numGridSteps - f), &s, x + o, y + o, z + o, &t, &t, &t, NULL, NULL, NULL);
                                    if (s < 0)
                                    {
                                        while (s < 0 && undo)
                                        {
                                            grid->sbcApply(x + o, y + o, z + o);
                                            s++;
                                        }
                                    }
                                    else
                                    {
                                        while (s > 0 && undo)
                                        {
                                            grid->sbcInvApply(x + o, y + o, z + o);
                                            s--;
                                        }
                                    }

                                    d[o] = (float)k;
                                    n = 1;
                                }
                            }
                        }
                    }
                }

                if (streakAngle->getValue() == 0.0)
                {
                    // no separation
                    int *ll, *vl;
                    ll = new int[numPart];
                    vl = new int[_k];

                    numVert = 0;
                    for (numLines = 0; numLines < numPart; numLines++)
                    {
                        ll[numLines] = numVert;
                        for (o = 0; o < numPathPart[numLines]; o++, numVert++)
                            vl[numVert] = numVert;
                    }

                    // generate covise objects
                    sprintf(bfr, "%s_%d", outPort1->getObjName(), i);
                    //lines = new coDoLines(bfr, o, (o/3)*2, o/3);
                    lines = new coDoLines(bfr, _k, x, y, z, _k, vl, numPart, ll);
                    sprintf(bfr, "%s_%d", outPort3->getObjName(), i);
                    dat = new coDoFloat(bfr, _k, d);
                    lines->addAttribute("COLOR", "blue");

                    delete[] vl;
                    delete[] ll;
                }
                else
                {
                    // compute number of lines we will get
                    float ax, ay, az, bx, by, bz;
                    int st = 0;
                    numLines = 0;
                    o = 0;
                    for (f = 0; f < numPart; f++)
                    {
                        numLines++;
                        st = 1; // line-mode
                        for (k = 1; k < numPathPart[f] - 2; k++)
                        {
                            // compute criteria
                            ax = x[o + k - 1] - x[o + k];
                            ay = y[o + k - 1] - y[o + k];
                            az = z[o + k - 1] - z[o + k];
                            bx = x[o + k + 1] - x[o + k];
                            by = y[o + k + 1] - y[o + k];
                            bz = z[o + k + 1] - z[o + k];

                            t = sqrt(ax * ax + ay * ay + az * az);
                            ax /= t;
                            ay /= t;
                            az /= t;
                            t = sqrt(bx * bx + by * by + bz * bz);
                            bx /= t;
                            by /= t;
                            bz /= t;

                            //t = acos(ax*bx + ay*by + az*bz);
                            t = ax * bx + ay * by + az * bz;

                            // check
                            if (t > sepAng) // >-0.8
                            {
                                // invalid segment

                                // toggle line-mode (st)
                                if (st)

                                    numLines++;
                                k++;
                            }
                            else
                            {
                                // valid segment
                            }
                        }
                        o += numPathPart[f];
                    }

                    // assemble the lines
                    int *ll, *vl;
                    ll = new int[numLines];
                    vl = new int[_k];

                    cerr << "  will generate " << numLines << " lines" << endl;

                    numVert = 0;
                    numLines = 0;
                    //o = 0;
                    for (f = 0; f < numPart; f++)
                    {
                        ll[numLines] = numVert;
                        numLines++;
                        o = numVert;
                        for (k = 1; k < numPathPart[f] - 2; k++)
                        {
                            // compute criteria
                            ax = x[o + k - 1] - x[o + k];
                            ay = y[o + k - 1] - y[o + k];
                            az = z[o + k - 1] - z[o + k];
                            bx = x[o + k + 1] - x[o + k];
                            by = y[o + k + 1] - y[o + k];
                            bz = z[o + k + 1] - z[o + k];

                            t = sqrt(ax * ax + ay * ay + az * az);
                            ax /= t;
                            ay /= t;
                            az /= t;
                            t = sqrt(bx * bx + by * by + bz * bz);
                            bx /= t;
                            by /= t;
                            bz /= t;

                            //t = acos(ax*bx + ay*by + az*bz);
                            t = ax * bx + ay * by + az * bz;

                            // check
                            if (t > sepAng) // >-0.8
                            {
                                k++;
                                ll[numLines] = numVert + k;
                                numLines++;
                            }
                        }

                        for (k = 0; k < numPathPart[f]; k++)
                        {
                            vl[numVert] = numVert;
                            numVert++;
                        }
                    }

                    cerr << "  generated: " << numLines << endl;

                    // generate covise objects
                    sprintf(bfr, "%s_%d", outPort1->getObjName(), i);
                    lines = new coDoLines(bfr, _k, x, y, z, numVert, vl, numLines, ll);
                    sprintf(bfr, "%s_%d", outPort3->getObjName(), i);
                    dat = new coDoFloat(bfr, _k, d);
                    lines->addAttribute("COLOR", "blue");

                    delete[] vl;
                    delete[] ll;
                }

                delete[] x;
                delete[] y;
                delete[] z;
                delete[] d;

                cerr << " generated " << numLines << endl;
            }
            else
            {
                lines = NULL;
                dat = NULL;
            }
            sObj[i] = lines;
            dObj[i] = dat;
        }

        delete[] numPathPart;
        delete[] offsPathPart;
        delete[] idxPathPart;

        cerr << "creating output object" << endl;

        rObj = new coDoSet(outPort1->getObjName(), sObj);
        sprintf(bfr, "1 %d", numGridSteps);
        rObj->addAttribute("TIMESTEP", bfr);

        switch (startStyle->getValue())
        {
        case PLANE:
            sprintf(bfr, "P%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            break;
        default:
            sprintf(bfr, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
        }
        rObj->addAttribute("FEEDBACK", bfr);

        outPort1->setCurrentObject(rObj);

        rObj = new coDoSet(outPort3->getObjName(), dObj);
        sprintf(bfr, "1 %d", numGridSteps);
        rObj->addAttribute("TIMESTEP", bfr);
        outPort3->setCurrentObject(rObj);
    }
    break;

    //=========================================================================================================
    case 4: // streakpoints
    {
        if (numGridSteps == 1)
        {
            buildOutput(nP);
            return;
        }

        char bfr[512];
        int undo = sbcUndoFlag->getValue();
        int numPart = nP / numGridSteps;

        int k, o, s, cycles, f; //, _k;

        coDistributedObject *rObj = NULL;
        coDistributedObject **sObj = NULL;
        coDistributedObject **dObj = NULL;
        coDoFloat *dat;
        coDoPoints *points;
        float *x, *y, *z, *d, u, v, w;

        sObj = new coDistributedObject *[numGridSteps + 1];
        dObj = new coDistributedObject *[numGridSteps + 1];
        sObj[numGridSteps] = NULL;
        dObj[numGridSteps] = NULL;

        int *numPathPart, *offsPathPart, *idxPathPart;
        numPathPart = new int[numPart];
        offsPathPart = new int[numPart];
        idxPathPart = new int[numPart];

        for (i = 0; i < numGridSteps; i++)
        {
            for (k = 0; k < numPart; k++)
            {
                numPathPart[k] = 0;
                offsPathPart[k] = 0;
            }

            o = 0;
            for (f = i; f >= 0; f--)
            {
                for (k = 0; k < numPart; k++)
                {
                    if (particles[k + f * numPart])
                    {
                        if (particles[k + f * numPart]->getNumSteps() > i - f)
                        {
                            o++;
                            numPathPart[k]++;
                        }
                    }
                }
            }
            n = 1;
            for (cycles = 0; n; cycles++)
            {
                n = 0;
                for (k = 0; k < numPart; k++)
                {
                    if (particles[k])
                    {
                        if (particles[k]->getNumSteps() > (numGridSteps * cycles) + i)
                        {
                            o++;
                            numPathPart[k]++;
                            n = 1;
                        }
                    }
                }
                for (k = 0; k < numPart; k++)
                {
                    for (f = numGridSteps - 1; f > 0; f--)
                    {
                        if (particles[k + f * numPart])
                        {
                            if (particles[k + f * numPart]->getNumSteps() > (numGridSteps * cycles) + i + (numGridSteps - f))
                            {
                                o++;
                                numPathPart[k]++;
                                n = 1;
                            }
                        }
                    }
                }
            }

            if (o > 0)
            {
                cerr << " found " << o << " points in step " << i << endl;

                // prepare on-the-fly sorting of the points
                s = 0;
                for (k = 0; k < numPart; k++)
                {
                    offsPathPart[k] = s;
                    s += numPathPart[k];
                    idxPathPart[k] = 0;
                }

                // generate covise objects
                sprintf(bfr, "%s_%d", outPort1->getObjName(), i);
                points = new coDoPoints(bfr, o);
                sprintf(bfr, "%s_%d", outPort3->getObjName(), i);
                dat = new coDoFloat(bfr, o);
                points->getAddresses(&x, &y, &z);
                points->addAttribute("COLOR", "blue");
                dat->getAddress(&d);

                // (beginning)
                for (f = i; f >= 0; f--)
                {
                    for (k = 0; k < numPart; k++)
                    {
                        if (particles[k + f * numPart])
                        {
                            if (particles[k + f * numPart]->getNumSteps() > i - f)
                            {
                                o = offsPathPart[k] + idxPathPart[k];
                                idxPathPart[k]++;

                                particles[k + f * numPart]->getWayPoint(i - f, &s, x + o, y + o, z + o, &u, &v, &w, NULL, NULL, NULL);
                                if (s < 0)
                                {
                                    while (s < 0 && undo)
                                    {
                                        grid->sbcApply(x + o, y + o, z + o);
                                        s++;
                                    }
                                }
                                else
                                {
                                    while (s > 0 && undo)
                                    {
                                        grid->sbcInvApply(x + o, y + o, z + o);
                                        s--;
                                    }
                                }
                                d[o] = (float)k;
                            }
                        }
                    }
                }

                // point, "others", point-cycled, "others"-cycled...etc etc
                n = 1;
                for (cycles = 0; n; cycles++)
                {
                    n = 0;
                    for (k = 0; k < numPart; k++)
                    {
                        if (particles[k])
                        {
                            if (particles[k]->getNumSteps() > (numGridSteps * cycles) + i)
                            {
                                // at least once
                                o = offsPathPart[k] + idxPathPart[k];
                                idxPathPart[k]++;

                                particles[k]->getWayPoint(i + (cycles * numGridSteps), &s, x + o, y + o, z + o, &u, &v, &w, NULL, NULL, NULL);
                                if (s < 0)
                                {
                                    while (s < 0 && undo)
                                    {
                                        grid->sbcApply(x + o, y + o, z + o);
                                        s++;
                                    }
                                }
                                else
                                {
                                    while (s > 0 && undo)
                                    {
                                        grid->sbcInvApply(x + o, y + o, z + o);
                                        s--;
                                    }
                                }

                                switch (whatDataOut->getValue())
                                {
                                case 0: // ID
                                    d[o] = (float)k;
                                    break;

                                /*
										case 1:								// velocity

										break;
										*/

                                case 2: // magnitude
                                    d[o] = sqrt(u * u + v * v + w * w);
                                    break;

                                default:
                                    d[o] = (float)k; // default is ID
                                }
                                n = 1;
                            }
                        }
                    }

                    for (k = 0; k < numPart; k++)
                    {
                        for (f = numGridSteps - 1; f > 0; f--)
                        {
                            if (particles[k + f * numPart])
                            {
                                if (particles[k + f * numPart]->getNumSteps() > (numGridSteps * cycles) + i + (numGridSteps - f))
                                {
                                    o = offsPathPart[k] + idxPathPart[k];
                                    idxPathPart[k]++;

                                    particles[k + f * numPart]->getWayPoint((numGridSteps * cycles) + i + (numGridSteps - f), &s, x + o, y + o, z + o, &u, &v, &w, NULL, NULL, NULL);
                                    if (s < 0)
                                    {
                                        while (s < 0 && undo)
                                        {
                                            grid->sbcApply(x + o, y + o, z + o);
                                            s++;
                                        }
                                    }
                                    else
                                    {
                                        while (s > 0 && undo)
                                        {
                                            grid->sbcInvApply(x + o, y + o, z + o);
                                            s--;
                                        }
                                    }

                                    switch (whatDataOut->getValue())
                                    {
                                    case 0: // ID
                                        d[o] = (float)k;
                                        break;

                                    /*
											case 1:								// velocity

											break;
											*/

                                    case 2: // magnitude
                                        d[o] = sqrt(u * u + v * v + w * w);
                                        break;

                                    default:
                                        d[o] = (float)k; // default is ID
                                    }
                                    n = 1;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                points = NULL;
                dat = NULL;
            }
            sObj[i] = points;
            dObj[i] = dat;
        }

        delete[] numPathPart;
        delete[] offsPathPart;
        delete[] idxPathPart;

        cerr << "creating output object" << endl;

        rObj = new coDoSet(outPort1->getObjName(), sObj);
        sprintf(bfr, "1 %d", numGridSteps);
        rObj->addAttribute("TIMESTEP", bfr);

        switch (startStyle->getValue())
        {
        case PLANE:
            sprintf(bfr, "P%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            break;
        default:
            sprintf(bfr, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
        }
        rObj->addAttribute("FEEDBACK", bfr);

        outPort1->setCurrentObject(rObj);

        rObj = new coDoSet(outPort3->getObjName(), dObj);
        sprintf(bfr, "1 %d", numGridSteps);
        rObj->addAttribute("TIMESTEP", bfr);
        outPort3->setCurrentObject(rObj);
    }
    break;

    //=========================================================================================================
    default: // points
    {

        if (numGridSteps == 1)
        {
            buildOutput(nP);
            return;
        }
        char bfr[512];
        float *x, *y, *z, *d, u, v, w;
        int k, o, s, cycles;
        int undo = sbcUndoFlag->getValue();

        coDistributedObject *rObj = NULL;
        coDistributedObject **sObj = NULL;
        coDoPoints *points;
        coDistributedObject **dObj = NULL;
        coDoFloat *dat;

        sObj = new coDistributedObject *[numGridSteps + 1];
        dObj = new coDistributedObject *[numGridSteps + 1];
        sObj[numGridSteps] = NULL;
        dObj[numGridSteps] = NULL;

        for (i = 0; i < numGridSteps; i++)
        {
            //cerr << "i = " << i << endl;

            // how many traces are still running ?!
            o = 0;
            for (k = 0; k < nP; k++)
            {
                if (particles[k])
                {
                    if (particles[k]->getNumSteps() > i)
                    {
                        // at least in the first cycle
                        o++;
                        // and maybe in others
                        for (cycles = 1; particles[k]->getNumSteps() > (numGridSteps * cycles) + i; cycles++)
                            ;
                        cycles--;
                        // hier ist der fehler.....hmmm
                        //cycles = (int)(particles[k]->getNumSteps()/(numGridSteps+1));

                        o += cycles;
                    }
                }
            }

            if (o > 0)
            {
                cerr << " found " << o << " points in step " << i << endl;

                sprintf(bfr, "%s_%d", outPort1->getObjName(), i);
                points = new coDoPoints(bfr, o);
                sprintf(bfr, "%s_%d", outPort3->getObjName(), i);
                dat = new coDoFloat(bfr, o);
                points->getAddresses(&x, &y, &z);
                points->addAttribute("COLOR", "blue");
                dat->getAddress(&d);

                o = 0;
                for (k = 0; k < nP; k++)
                {
                    if (particles[k])
                    {
                        if (particles[k]->getNumSteps() > i)
                        {
                            // at least once
                            particles[k]->getWayPoint(i, &s, x + o, y + o, z + o, &u, &v, &w, NULL, NULL, NULL);
                            if (s < 0)
                            {
                                while (s < 0 && undo)
                                {
                                    grid->sbcApply(x + o, y + o, z + o);
                                    s++;
                                }
                            }
                            else
                            {
                                while (s > 0 && undo)
                                {
                                    grid->sbcInvApply(x + o, y + o, z + o);
                                    s--;
                                }
                            }

                            switch (whatDataOut->getValue())
                            {
                            case 0: // ID
                                d[o] = (float)k;
                                break;

                            /*
									case 1:								// velocity

									break;
									*/

                            case 2: // magnitude
                                d[o] = sqrt(u * u + v * v + w * w);
                                break;

                            default:
                                d[o] = (float)k; // default is ID
                            }

                            o++;

                            // and maybe later ;)
                            //cycles = (int)(particles[k]->getNumSteps()/(numGridSteps+1));

                            for (n = 1; particles[k]->getNumSteps() > (numGridSteps * n) + i; n++)
                            {
                                particles[k]->getWayPoint(i + (n * numGridSteps), &s, x + o, y + o, z + o, &u, &v, &w, NULL, NULL, NULL);
                                while (s > 0 && undo)
                                {
                                    grid->sbcInvApply(x + o, y + o, z + o);
                                    s--;
                                }
                                switch (whatDataOut->getValue())
                                {
                                case 0: // ID
                                    d[o] = (float)k;
                                    break;

                                /*
										case 1:								// velocity

										break;
										*/

                                case 2: // magnitude
                                    d[o] = sqrt(u * u + v * v + w * w);
                                    break;

                                default:
                                    d[o] = (float)k; // default is ID
                                }
                                o++;
                            }
                        }
                    }
                }

                cerr << " " << o << " points generated" << endl;
            }
            else
            {
                points = NULL;
                dat = NULL;
            }
            sObj[i] = points;
            dObj[i] = dat;
        }

        cerr << "creating output object" << endl;

        rObj = new coDoSet(outPort1->getObjName(), sObj);
        sprintf(bfr, "1 %d", numGridSteps);
        rObj->addAttribute("TIMESTEP", bfr);

        switch (startStyle->getValue())
        {
        case PLANE:
            sprintf(bfr, "P%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
            break;
        default:
            sprintf(bfr, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
        }
        rObj->addAttribute("FEEDBACK", bfr);

        outPort1->setCurrentObject(rObj);

        rObj = new coDoSet(outPort3->getObjName(), dObj);
        sprintf(bfr, "1 %d", numGridSteps);
        rObj->addAttribute("TIMESTEP", bfr);
        outPort3->setCurrentObject(rObj);
    }
    }

    cerr << "done" << endl;

    // done
    return;
}

void Tracer::buildOutput(int nP)
{
    int i, n, k, o, s;
    char bfr[512];
    float *x, *y, *z, *d, u, v, w;

    n = 0;
    for (i = 0; i < nP; i++)
    {
        if (particles[i])
            if (particles[i]->getNumSteps() > n)
                n = particles[i]->getNumSteps();
    }

    cerr << "### Tracer::buildOutput,   " << nP << " particles,  " << n << " steps" << endl;

    coDistributedObject *rObj = NULL;
    coDistributedObject **sObj = NULL;
    coDoPoints *points;
    coDistributedObject **dObj = NULL;
    coDoFloat *dat;

    sObj = new coDistributedObject *[n + 1];
    dObj = new coDistributedObject *[n + 1];
    sObj[n] = NULL;
    dObj[n] = NULL;

    for (i = 0; i < n; i++)
    {
        //cerr << "i = " << i << endl;

        o = 0;
        for (k = 0; k < nP; k++)
            if (particles[k])
                if (particles[k]->getNumSteps() > i)
                    o++;
        if (o > 0)
        {
            sprintf(bfr, "%s_%d", outPort1->getObjName(), i);
            points = new coDoPoints(bfr, o);
            sprintf(bfr, "%s_%d", outPort3->getObjName(), i);
            dat = new coDoFloat(bfr, o);
            points->getAddresses(&x, &y, &z);
            points->addAttribute("COLOR", "blue");
            dat->getAddress(&d);
            o = 0;
            for (k = 0; k < nP; k++)
            {
                if (particles[k])
                {
                    if (particles[k]->getNumSteps() > i)
                    {
                        particles[k]->getWayPoint(i, &s, x + o, y + o, z + o, &u, &v, &w, NULL, NULL, NULL);
                        switch (whatDataOut->getValue())
                        {
                        case 0: // ID
                            d[o] = (float)k;
                            break;

                        /*
							case 1:								// velocity

							break;
							*/

                        case 2: // magnitude
                            d[o] = sqrt(u * u + v * v + w * w);
                            break;

                        default:
                            d[o] = (float)k; // default is ID
                        }
                        o++;
                    }
                }
            }
        }
        else
        {
            points = NULL;
            dat = NULL;
        }
        sObj[i] = points;
        dObj[i] = dat;
    }

    cerr << "creating output object" << endl;

    rObj = new coDoSet(outPort1->getObjName(), sObj);
    sprintf(bfr, "1 %d", n);
    rObj->addAttribute("TIMESTEP", bfr);

    sprintf(bfr, "T%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
    rObj->addAttribute("FEEDBACK", bfr);

    outPort1->setCurrentObject(rObj);

    rObj = new coDoSet(outPort3->getObjName(), dObj);
    sprintf(bfr, "1 %d", n);
    rObj->addAttribute("TIMESTEP", bfr);
    outPort3->setCurrentObject(rObj);

    //cerr << "cleaning up" << endl;

    cerr << "done" << endl;

    // done
    return;
}

//===============================================================================================
//===============================================================================================
//===============================================================================================
//===============================================================================================

WristWatch::WristWatch()
{
}

WristWatch::~WristWatch()
{
}

void WristWatch::start()
{
#ifndef _WIN32
    gettimeofday(&myClock, NULL);
#endif
    return;
}

void WristWatch::stop(char *s)
{
#ifndef _WIN32
    timeval now;
    float dur;
    gettimeofday(&now, NULL);
    dur = ((float)(now.tv_sec - myClock.tv_sec)) + ((float)(now.tv_usec - myClock.tv_usec)) / 1000000.0;
    Covise::sendInfo("%s took %6.3f seconds.", s, dur);
#endif

    return;
}

MODULE_MAIN(UnderDev, Tracer)
