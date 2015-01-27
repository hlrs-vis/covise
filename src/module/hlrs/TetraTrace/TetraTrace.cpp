/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <appl/ApplInterface.h>
#include "TetraTrace.h"
#include "TetraGrid.h"
#include "trace.h"

#include <util/coviseCompat.h>
#include <do/coDoSet.h>

int main(int argc, char *argv[])
{
    // init
    TetraTrace *application = new TetraTrace(argc, argv);

    // and back to covise
    application->run();

    // done
    return 1;
}

TetraTrace::TetraTrace(int argc, char *argv[])
{
    Covise::set_module_description("perform particle-trace on tetrahedra-grids");

    Covise::add_port(INPUT_PORT, "gridIn", "UnstructuredGrid", "tetrahedra-grid");
    Covise::add_port(INPUT_PORT, "velIn", "Vec3", "velocity input");
    Covise::add_port(INPUT_PORT, "volIn", "Float", "volumes of tetrahedras");
    Covise::add_port(INPUT_PORT, "neighborIn", "Float", "neighbors of tetrahedras");

    Covise::add_port(OUTPUT_PORT, "traceOut", "Points|Polygons|Lines|TriangleStrips", "computed trace");
    Covise::add_port(OUTPUT_PORT, "dataOut", "Float|Vec3", "data to be mapped on the trace");

    Covise::add_port(PARIN, "startpoint1", "FloatVector", "...");
    Covise::set_port_default("startpoint1", "1.0 1.0 1.0");

    Covise::add_port(PARIN, "startpoint2", "FloatVector", "...");
    Covise::set_port_default("startpoint2", "1.0 2.0 1.0");

    Covise::add_port(PARIN, "normal", "FloatVector", "...");
    Covise::set_port_default("normal", "0.0 0.0 1.0");

    Covise::add_port(PARIN, "direction", "FloatVector", "...");
    Covise::set_port_default("direction", "1.0 0.0 0.0");

    Covise::add_port(PARIN, "numStart", "IntScalar", "number of traces to start");
    Covise::set_port_default("numStart", "10");

    Covise::add_port(PARIN, "startStep", "IntScalar", "initial timestep (transient only)");
    Covise::set_port_default("startStep", "1");

    Covise::add_port(PARIN, "whatOut", "Choice", "what data should we compute");
    Covise::set_port_default("whatOut", "1 number velocity magnitude");

    Covise::add_port(PARIN, "startStyle", "Choice", "how to compute starting-points");
    Covise::set_port_default("startStyle", "1 line plane sphere box");

    Covise::add_port(PARIN, "traceStyle", "Choice", "how to output the trace");
    Covise::set_port_default("traceStyle", "1 points lines easymesh mesh fader");

    Covise::add_port(PARIN, "numSteps", "IntScalar", "number of steps to compute");
    Covise::set_port_default("numSteps", "100");

    Covise::add_port(PARIN, "stepDuration", "FloatScalar", "duration of each step");
    Covise::set_port_default("stepDuration", "0.01");

    Covise::add_port(PARIN, "numNodes", "IntScalar", "number of nodes/processors to use");
    Covise::set_port_default("numNodes", "1");

    Covise::add_port(PARIN, "multiProcMode", "Choice", "which multi-processing mode to use");
    Covise::set_port_default("multiProcMode", "1 none SMP MMP");

    Covise::add_port(PARIN, "searchMode", "Choice", "which searching-algorithm to use for startcells");
    Covise::set_port_default("searchMode", "1 quick save");

    Covise::init(argc, argv);

    const char *in_names[] = { "gridIn", "velIn", "volIn", "neighborIn", NULL };
    const char *out_names[] = { "traceOut", "dataOut", NULL };

    setPortNames(in_names, out_names);
    setComputeTimesteps(0);

    setCallbacks();

    return;
};

TetraTrace::~TetraTrace()
{
    // nothing left to do
}

void TetraTrace::getParameters()
{
    int i;

    for (i = 0; i < 3; i++)
    {
        Covise::get_vector_param("startpoint1", i, &startpoint1[i]);
        Covise::get_vector_param("startpoint2", i, &startpoint2[i]);
        Covise::get_vector_param("normal", i, &startNormal[i]);
        Covise::get_vector_param("direction", i, &startDirection[i]);
    }

    Covise::get_scalar_param("numStart", &numStart);
    Covise::get_scalar_param("startStep", &startStep);
    Covise::get_choice_param("whatOut", &whatOut);
    Covise::get_choice_param("startStyle", &startStyle);
    Covise::get_choice_param("traceStyle", &traceStyle);
    Covise::get_scalar_param("numSteps", &numSteps);
    Covise::get_scalar_param("stepDuration", &stepDuration);
    Covise::get_scalar_param("numNodes", &numNodes);
    Covise::get_choice_param("multiProcMode", &multiProcMode);
    Covise::get_choice_param("searchMode", &searchMode);

    // check for unsupported options

    // check for nonsense input
    if (numStart <= 1)
        numStart = 10;

    if (numSteps < 1)
        numSteps = 1;

    if (stepDuration <= 0.0)
        stepDuration = 0.01f;

    if (numNodes < 1)
        numNodes = 1;

    if (startStep < 1)
        startStep = 1;
    startStep--;

    if (traceStyle == 5)
        whatOut = 2;

    // done
    return;
}

int TetraTrace::getInput(const coDistributedObject **objIn)
{
    const char *dataType = (objIn[0])->getType();
    if (!strcmp(dataType, "SETELE"))
    {
        // we have a set -> transient case
        const coDistributedObject *const *setIn0, *const *setIn1, *const *setIn2, *const *setIn3;
        int i;

        setIn0 = ((const coDoSet *)objIn[0])->getAllElements(&numGrids);
        setIn1 = ((const coDoSet *)objIn[1])->getAllElements(&i);
        setIn2 = ((const coDoSet *)objIn[2])->getAllElements(&i);
        setIn3 = ((const coDoSet *)objIn[3])->getAllElements(&i);

        traceGrid = new TetraGrid *[numGrids];
        for (i = 0; i < numGrids; i++)
            traceGrid[i] = new TetraGrid((coDoUnstructuredGrid *)setIn0[i],
                                         (coDoVec3 *)setIn1[i],
                                         (coDoFloat *)setIn2[i],
                                         (coDoFloat *)setIn3[i], searchMode, numStart);
    }
    else
    {
        // stationary case
        numGrids = 1;
        traceGrid = new TetraGrid *[1];
        traceGrid[0] = new TetraGrid((const coDoUnstructuredGrid *)objIn[0],
                                     (coDoVec3 *)objIn[1],
                                     (coDoFloat *)objIn[2],
                                     (coDoFloat *)objIn[3], searchMode, numStart);
    }

    // check for nonsens input
    if (startStep > numGrids - 1)
        startStep = 0;

    // done
    return (1);
}

coDistributedObject **TetraTrace::compute(const coDistributedObject **objIn, char **outNames)
{
    coDistributedObject **returnObject = NULL;
    trace *tr = NULL;

    int i;

    // get parameters
    getParameters();

    // and input
    getInput(objIn);

    // perform the trace
    tr = new trace();
    tr->setParameters(startpoint1, startpoint2, startNormal, startDirection,
                      numStart, whatOut, startStyle, traceStyle, numSteps,
                      stepDuration, numNodes, multiProcMode, startStep, searchMode);
    tr->setInput(numGrids, traceGrid);
    returnObject = tr->run(outNames);

    // clean up
    delete tr;
    for (i = 0; i < numGrids; i++)
        delete traceGrid[i];
    delete[] traceGrid;

    // done
    return (returnObject);
}
