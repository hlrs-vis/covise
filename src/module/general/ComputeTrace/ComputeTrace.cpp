/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include <do/coDoSpheres.h>
#include <appl/ApplInterface.h>
#include "ComputeTrace.h"
#include <vector>
#include <string>

#include <api/coFeedback.h>

static const char *animValues[] = { "Off", "Keep direction", "Look at fixed point", "Roller coaster", "Look back" };
enum AnimValues
{
    AnimOff = 0,
    AnimKeepDirection,
    AnimFocusFixedPoint,
    AnimRollerCoaster,
    AnimLookBack,
    AnimNumValues
};

ComputeTrace::ComputeTrace(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Creates traces from points timestep dependent")
    , m_firsttime(true)
{
    //the timesteps shall NOT be handled automatically by the coSimpleModule class
    setComputeTimesteps(1);

    /*Fed in points*/
    p_pointsIn = addInputPort("GridIn0", "Points|Spheres", "a set of spheres containing the particle/s to be traced over time");
    p_dataIn = addInputPort("DataIn0", "Float|Byte|Int|Vec2|Vec3|RGBA|Mat3|Tensor", "data mapped associated with spheres");
    p_dataIn->setRequired(false);
    p_IDIn = addInputPort("IDIn0", "Int", "ID of each atom");
    p_IDIn->setRequired(false);
    /*Boolean that specifies if a particle should be traced or not*/
    p_traceParticle = addBooleanParam("traceParticle", "set if particle should be traced");
    p_traceParticle->setValue(1);
    /*Integer that specifies the particle to be traced in Module Parameter window*/
    p_particle = addStringParam("selection", "ranges of selected set elements");
    p_particle->setValue("1-1");
    p_maxParticleNumber = addInt32Param("maxParticleNum", "maximum number of particles to trace");
    p_maxParticleNumber->setValue(100);
    /*Integer that specifies the starting point*/
    p_start = addIntSliderParam("start", "Timestep at which the tracing should be started");
    p_start->setValue(0, 0, 0);
    /*Integer that specifies the ending point*/
    p_stop = addIntSliderParam("stop", "Timestep at which the tracing should be stopped");
    p_stop->setValue(0, 0, 0);
    /* Boolean that specifies if the bounding box leaving should be regarded */
    p_regardInterrupt = addBooleanParam("LeavingBoundingBox", "set if leaving bounding box should be taken into account");
    p_regardInterrupt->setValue(0);
    p_animate = addBooleanParam("animate", "disable for static trace");
    p_animate->setValue(1);
    /* 3 values specifying the x, y and z dimension of the bounding box */
    p_boundingBoxDimensions = addFloatVectorParam("BoundingBoxDimensions", "x, y, z dimensions of the bounding box");
    p_boundingBoxDimensions->setValue(0, 0, 0);
    /*Output should be a line*/
    p_traceOut = addOutputPort("GridOut0", "Lines", "Trace of a specified particle");
    /*unique index per line mappable as color attribute*/
    p_indexOut = addOutputPort("DataOut0", "Float", "unique index for every specified particle");
    /*fade out value per timestep mappable as color attribute*/
    p_fadingOut = addOutputPort("DataOut1", "Float", "fade out value for per vertex coloring of lines over time");
    p_dataOut = addOutputPort("DataOut2", "Float|Byte|Int|Vec2|Vec3|RGBA|Mat3|Tensor", "data mapped to lines");

    p_dataOut->setDependencyPort(p_dataIn);
    IDs = NULL;

    assert(sizeof(animValues) / sizeof(animValues[0]) == AnimNumValues);
    p_animateViewer = addChoiceParam("animateViewer", "Animate Viewer");
    p_animateViewer->setValue(AnimNumValues, animValues, AnimOff);

    p_animLookAt = addFloatVectorParam("animLookAt", "Animated viewer looks at this point");
    p_animLookAt->setValue(0., 0., 0.);
}

void ComputeTrace::param(const char *name, bool /*inMapLoading*/)
{
    if (strcmp(name, p_start->getName()) == 0)
    {
        if (p_start->getValue() > p_stop->getValue())
        {
            //sendWarning("start is larger than stop, start is set to stop");
            p_stop->setValue(p_start->getValue());
        }
    }
    if (strcmp(name, p_stop->getName()) == 0)
    {
        if (p_stop->getValue() < p_start->getValue())
        {
            //sendWarning("stop is smaller than start, stop is set to start");
            p_start->setValue(p_stop->getValue());
        }
    }
}

//creates a char array containing a basis and 1 or 2 indices added
//separated by underscores
std::string ComputeTrace::createIndexedName(const char *basis, int i, int j = -1)
{
    stringstream ss;
    ss << basis << "_" << i;
    if (j > -1)
    {
        ss << "_" << j;
    }
    return ss.str();
}

int ComputeTrace::compute(const char *)
{
    //input port object
    const coDistributedObject *obj = p_pointsIn->getCurrentObject();

    if (!obj)
    {
        //no input, no output
        sendError("Did not receive object at port '%s'", p_pointsIn->getName());
        return FAIL;
    }

    const coDoSet *IDset = dynamic_cast<const coDoSet *>(p_IDIn->getCurrentObject());
    if (IDset)
    {
        delete IDs;
        IDs = new int *[IDset->getNumElements()];
        for (int i = 0; i < IDset->getNumElements(); i++)
        {
            const coDoInt *IDObj = dynamic_cast<const coDoInt *>(IDset->getElement(i));
            if (IDObj)
            {
                IDObj->getAddress(&IDs[i]);
            }
            else
            {
                delete IDs;
                IDs = NULL;
                break;
            }
        }
    }
    if (const coDoSet *set = dynamic_cast<const coDoSet *>(obj))
    {
        if (set->getAttribute("TIMESTEP"))
        {
            // the TIMESTEP attribute arguments are not correct on many datasets, use 0 to numElems instead
            // the module fails if m_timestepStart>0
            /*int ret = sscanf(set->getAttribute("TIMESTEP"),"%d %d", &m_timestepStart, &m_timestepStop);
         if(ret == 0)
         {
            sendError("couldn't successfully read TIMESTEP arguments");
            return FAIL;
         }*/
            m_timestepStart = 0;
            m_timestepStop = set->getNumElements() - 1;

            if (m_firsttime)
            {
                m_firsttime = false;
                p_start->setValue(m_timestepStart, m_timestepStop, m_timestepStart);
                p_stop->setValue(m_timestepStart, m_timestepStop, m_timestepStop);
            }
            else
            {
                p_start->setValue(p_start->getValue(), m_timestepStop, m_timestepStart);
                p_stop->setValue(p_stop->getValue(), m_timestepStop, m_timestepStop);
            }
        }

        /* Creating the output object.
       * The TIMESTEP attribute is set automatically since this class
       * is derived of the coSimpleModule class*/
        if (p_traceOut->getObjName())
        {
            std::string name;
            // test if input objects are of coDoSpheres* type
            if (dynamic_cast<const coDoSpheres *>(set->getElement(0)))
            {
                // creating a string that contains all names of
                // all coDoSpheres objects separated by newline escape sequence
                int numElem = set->getNumElements();
                for (int i = 0; i < numElem; i++)
                {
                    name += (set->getElement(i))->getName();
                    name += "\n";
                }
            }

            //Compensating input errors caused by the user
            m_start = p_start->getValue();
            m_stop = p_stop->getValue();

            if (m_start < m_timestepStart)
            {
                sendWarning("start is smaller than the beginning of the timestep; start is set to min_timestep");
                m_start = m_timestepStart;
                p_start->setValue(m_start);
            }

            if (m_stop > m_timestepStop)
            {
                sendWarning("stop is larger than the maximum of timesteps; stop is set to max_timestep");
                m_stop = m_timestepStop;
                p_stop->setValue(m_stop);
            }

            // getting the selected particle string and
            // create the integer vector holding the element
            // indices with coRestraint
            m_particleIndices.clear();
            //fprintf(stderr, "\np_particle->getValue: %s\n", p_particle->getValue() );
            m_particleIndices.add(p_particle->getValue());
            m_particleSelection = m_particleIndices.getValues();

            // setting sorted, newly organized and formatted
            // string as particle string value
            std::string cleanedupString = m_particleIndices.getRestraintString();
            p_particle->setValue(cleanedupString.c_str());

            //Getting number of particles in the first timestep assuming that every
            //new timestep holds the same number of particles
            int no_of_points = 0;
            float min[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
            float max[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            if (const coDoCoordinates *coord = dynamic_cast<const coDoCoordinates *>(set->getElement(0)))
            {
                no_of_points = coord->getNumElements();
                for (int i = 0; i < no_of_points; ++i)
                {
                    float x, y, z;
                    coord->getPointCoordinates(i, &x, &y, &z);
                    if (x > max[0])
                        max[0] = x;
                    if (y > max[1])
                        max[1] = y;
                    if (z > max[2])
                        max[2] = z;
                    if (x < min[0])
                        min[0] = x;
                    if (y < min[1])
                        min[1] = y;
                    if (z < min[2])
                        min[2] = z;
                }
                p_boundingBoxDimensions->setValue(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
            }
            else if (dynamic_cast<const coDoTriangleStrips *>(set->getElement(0)))
            {
                sendError("Use Manual CPU Billboards instead of Polygons.");
                return FAIL;
            }
            else
            {
                sendError("wrong input type at input port. Type must be either Spheres or Points.");
                return FAIL;
            }

            //test if specified particles smaller than the maximum number of particles
            if (IDs == NULL)
            {
                if (!m_particleSelection.empty() && m_particleSelection.back() >= no_of_points)
                {
                    stringstream range;
                    range << "0-" << no_of_points - 1;
                    sendWarning("The particle index you specified is out of range. Particles out of range will be removed. Valid range is %s", range.str().c_str());
                    while (!m_particleSelection.empty() && m_particleSelection.back() >= no_of_points)
                    {
                        m_particleSelection.pop_back();
                    }
                    std::string modifiedString = m_particleIndices.getRestraintString(m_particleSelection);
                    p_particle->setValue(modifiedString.c_str());
                }

                if (m_particleSelection.size() > p_maxParticleNumber->getValue())
                {
                    while (m_particleSelection.size() > p_maxParticleNumber->getValue())
                        m_particleSelection.pop_back();
                }
            }

            //structure that holds the output components
            coDistributedObject **traceLines = NULL;

            // depending on the selection string create the
            // traces or a NULL object that holds NULL-line elements
            // for every timestep
            bool empty = false;
            if (!p_traceParticle->getValue() || m_particleSelection.empty())
            {
                traceLines = computeDynamicTrace(set, false);
                empty = true;
                fprintf(stderr, "empty: traceLines=%p\n", traceLines);
            }
            else if (p_animate->getValue())
            {
                traceLines = computeDynamicTrace(set, true);
            }
            else
            {
                traceLines = new coDistributedObject *[1];
                traceLines[0] = computeStaticTrace(std::string(p_traceOut->getObjName()), set);
            }

            // creating the output
            coDistributedObject *traces = (p_animate->getValue() || empty)
                                              ? new coDoSet(p_traceOut->getObjName(), traceLines)
                                              : traceLines[0];
            delete[] traceLines;

            if (dynamic_cast<const coDoSpheres *>(set->getElement(0)))
            {
                // creating the feedback object needed for
                // openCOVER<->COVISE interaction
                coFeedback feedback("PickSphere");
                feedback.addPara(p_start);
                feedback.addPara(p_stop);
                feedback.addPara(p_particle);
                feedback.addPara(p_traceParticle);
                feedback.addPara(p_regardInterrupt);

                // all coDoSphere object names are attached to the trace lines
                traces->addAttribute("PICKSPHERE", name.c_str());

                // viewpoint animation parameters
                feedback.addPara(p_animateViewer);
                feedback.addPara(p_animLookAt);

                // attaching the feedback object to trace lines
                feedback.apply(traces);
            }

            // setting the output object
            p_traceOut->setCurrentObject(traces);
        }
        else
        {
            fprintf(stderr, "Covise::getObjName failed\n");
            return FAIL;
        }

        if (p_indexOut->getObjName())
        {
            coDistributedObject *indices = computeFloats(returnIndex, p_indexOut->getObjName());
            p_indexOut->setCurrentObject(indices);
        }
        else
        {
            fprintf(stderr, "Covise::getObjName failed\n");
            return FAIL;
        }

        if (p_fadingOut->getObjName())
        {
            coDistributedObject *fading = computeFloats(fade, p_fadingOut->getObjName());
            p_fadingOut->setCurrentObject(fading);
        }
        else
        {
            fprintf(stderr, "Covise::getObjName failed\n");
            return FAIL;
        }

        if (p_dataOut->getObjName())
        {
            if (const coDoSet *set = dynamic_cast<const coDoSet *>(p_dataIn->getCurrentObject()))
                p_dataOut->setCurrentObject(extractData(p_dataOut->getObjName(), set, p_animate->getValue()));
            else if (p_dataIn->getCurrentObject())
                fprintf(stderr, "need a set of data");
        }
        else
        {
            fprintf(stderr, "Covise::getObjName failed\n");
            return FAIL;
        }
    }
    else
    {
        sendInfo("need a set object with time steps");
        return FAIL;
    }
    return SUCCESS;
}

float
ComputeTrace::fade(float /*dummy*/, int exp)
{
    return pow(0.98, exp);
}

float
ComputeTrace::returnIndex(float index, int /*dummy*/)
{
    return index;
}

coDistributedObject *ComputeTrace::computeFloats(float (*alternative)(float, int), const char *objName)
{
    const size_t steps = 1 + m_timestepStop - m_timestepStart;
    if (p_animate->getValue())
    {
        std::vector<coDistributedObject *> traceFade(steps);
        coDoFloat *nullFloat = new coDoFloat(createIndexedName(objName, 0), 0, 0);

        for (int i = m_timestepStart; i < m_start; i++)
        {
            if (i > m_timestepStart)
                nullFloat->incRefCount(); //reuse of nullLine
            traceFade[i] = nullFloat;
        }

        //creating a temporary variable that holds all traces of the traced particles
        //of one timestep. These objects will be put together in a coDoSet.
        std::vector<coDistributedObject *> floatSet(m_particleSelection.size());

        //successive creation of the traces between start and stop
        for (size_t i = m_start; i <= m_stop; i++)
        {
            for (size_t particle = 0; particle < m_particleSelection.size(); particle++)
            {
                floatSet[particle] = new coDoFloat(createIndexedName(objName, i, particle).c_str(), i + 1 - m_start);
                float *data = ((coDoFloat *)floatSet[particle])->getAddress();
                for (size_t j = 0; j < i + 1 - m_start; j++)
                {
                    data[j] = alternative(particle / (float)m_particleSelection.size(), (i - m_start - j));
                }
            }
            //creating the line object set for the current timestep
            traceFade[i] = new coDoSet(createIndexedName(objName, i).c_str(), floatSet.size(), &floatSet[0]);
        }
        //last drawn line will be visible for every timestep after stop
        for (int i = m_stop + 1; i <= m_timestepStop; i++)
        {
            traceFade[i] = traceFade[i - 1];
            traceFade[i]->incRefCount();
        }
        return new coDoSet(objName, traceFade.size(), &traceFade[0]);
    }
    else
    {
        coDoFloat *floats = new coDoFloat(objName, steps*m_particleSelection.size());
        float *data = floats->getAddress();
        for (size_t particle = 0; particle < m_particleSelection.size(); particle++)
        {
            for (size_t j = 0; j < steps; j++)
            {
                size_t idx = m_particleSelection.size()*particle+j;
                data[idx] = alternative(particle / (float)m_particleSelection.size(), j-m_timestepStart);
            }
        }
        return floats;
    }
}

int ComputeTrace::getIndex(int t, int numIDs, int ID)
{
    for (int i = 0; i < numIDs; i++)
    {
        if (IDs[t][i] == ID)
            return i;
    }
    return -1;
}
coDoLines *ComputeTrace::computeStaticTrace(const std::string &name, const coDoSet *set)
{
    const int steps = m_timestepStop - m_timestepStart + 1;
    coDoLines *lines = new coDoLines(name, steps * m_particleSelection.size(),
                                     steps * m_particleSelection.size(), m_particleSelection.size());
    float *x, *y, *z;
    int *v_l, *l_l;
    lines->getAddresses(&x, &y, &z, &v_l, &l_l);

    for (int particle = 0; particle < m_particleSelection.size(); particle++)
        l_l[particle] = particle * steps;

    for (int t = m_timestepStart; t <= m_timestepStop; ++t)
    {
        if (const coDoCoordinates *coord = dynamic_cast<const coDoCoordinates *>(set->getElement(t)))

            for (int particle = 0; particle < m_particleSelection.size(); particle++)
            {
                int i = steps * particle + t - m_timestepStart;
                v_l[i] = i;
                if (IDs)
                {
                    int idx = getIndex(t, coord->getNumElements(), m_particleSelection[particle]);
                    if (idx >= 0)
                        coord->getPointCoordinates(idx, &x[i], &y[i], &z[i]);
                    else
                    {
                        if (t > 0)
                        {
                            int oldi = steps * particle + (t - 1) - m_timestepStart;
                            x[i] = x[oldi];
                            y[i] = y[oldi];
                            z[i] = z[oldi];
                        }
                        else
                        {
                            x[i] = 0;
                            y[i] = 0;
                            z[i] = 0;
                        }
                    }
                }
                else
                    coord->getPointCoordinates(m_particleSelection[particle], &x[i], &y[i], &z[i]);
            }
    }

    return lines;
}

coDistributedObject **ComputeTrace::computeDynamicTrace(const coDoSet *set, bool compute)
{
    //outPort object name
    const char *nameObjOut = p_traceOut->getObjName();

    //null line representation
    coDoLines *nullLine = new coDoLines(createIndexedName(nameObjOut, 0), 0, 0, 0);

    //Creating NULL terminated array that holds each line for each timestep
    //structure that holds the output components
    coDistributedObject **traceLines = new coDistributedObject *[m_timestepStop - m_timestepStart + 2];
    traceLines[m_timestepStop - m_timestepStart + 1] = NULL;

    if (!compute)
    {
        for (int i = m_timestepStart; i <= m_timestepStop; i++)
        {
            if (i > m_timestepStart)
                nullLine->incRefCount(); //reuse of nullLine
            traceLines[i] = nullLine;
        }
        return traceLines;
    }

    //NULL representation of a line for each timestep where no trace should be drawn
    for (int i = m_timestepStart; i < m_start; i++)
    {
        if (i > m_timestepStart)
            nullLine->incRefCount(); //reuse of nullLine
        traceLines[i] = nullLine;
    }

    //creating a temporary variable that holds all traces of the traced particles
    //of one timestep. These objects will be put together in a coDoSet.
    coDistributedObject **lineSet;
    lineSet = new coDistributedObject *[m_particleSelection.size() + 1];
    lineSet[m_particleSelection.size()] = NULL;

    //successive creation of the traces between start and stop
    for (int i = m_start; i <= m_stop; i++)
    {
        //point set of current timestep
        if (const coDoCoordinates *actualCoordinates = dynamic_cast<const coDoCoordinates *>(set->getElement(i)))
        {
            for (int particle = 0; particle < m_particleSelection.size(); particle++)
            {
                if (i == m_start)
                {
                    lineSet[particle] = new coDoLines(createIndexedName(nameObjOut, i, particle), 0, 0, 1);
                }

                //getting the point of interest
                float x, y, z;

                if (IDs)
                {
                    int idx = getIndex(i, actualCoordinates->getNumElements(), m_particleSelection[particle]);
                    if (idx >= 0)
                        actualCoordinates->getPointCoordinates(idx, &x, &y, &z);
                    else
                    {
                        if (i > 0)
                        {
                            coDoLines *oldLine = dynamic_cast<coDoLines *>(lineSet[particle]);
                            float *ox, *oy, *oz;
                            int *ocorners, *olines;
                            oldLine->getAddresses(&ox, &oy, &oz, &ocorners, &olines);

                            int oldi = oldLine->getNumPoints() - 1;
                            x = ox[oldi];
                            y = oy[oldi];
                            z = oz[oldi];
                        }
                        else
                        { // TODO, if atom appears later
                            x = 0;
                            y = 0;
                            z = 0;
                        }
                    }
                }
                else
                    actualCoordinates->getPointCoordinates(m_particleSelection[particle], &x, &y, &z);

                //compute the actual line of the actual particle
                lineSet[particle] = ComputeTrace::computeCurrentTrace(createIndexedName(nameObjOut, i, particle),
                                                                      dynamic_cast<coDoLines *>(lineSet[particle]), i - m_start, x, y, z);
            }
            //creating the line object set for the current timestep
            traceLines[i] = new coDoSet(createIndexedName(nameObjOut, i), lineSet);
        }
    }
    //last drawn line will be visible for every timestep after stop
    for (int i = m_stop + 1; i <= m_timestepStop; i++)
    {
        traceLines[i] = traceLines[i - 1];
        traceLines[i]->incRefCount();
    }
    return traceLines;
}

coDoLines *ComputeTrace::computeCurrentTrace(const std::string &name, coDoLines *old_trace, int i, float x_new, float y_new, float z_new)
{
    //printf("name in computeCurrentTrace: %s\n", name.c_str());
    // interrupt flag set if particle leaves bounding box
    bool interrupt = false;

    //Get the old values of the old coDoLines object
    float *ox, *oy, *oz;
    int *ocorners, *olines;
    old_trace->getAddresses(&ox, &oy, &oz, &ocorners, &olines);

    // calculate if the traced object leaves bounding box

    // flag if  particle leaving the bounding should be
    // taken into account in line representation
    bool regard_interrupt = p_regardInterrupt->getValue();
    if (regard_interrupt)
    {
        if (i > 0)
        {
            float x_diff = ox[i - 1] - x_new;
            float y_diff = oy[i - 1] - y_new;
            float z_diff = oz[i - 1] - z_new;

            // parameter specifying the width of a quader (bounding box)
            float param_x, param_y, param_z;
            p_boundingBoxDimensions->getValue(param_x, param_y, param_z);
            // particle leaves bounding box
            if (fabs(x_diff) > param_x * .5 || fabs(y_diff) > param_y * .5 || fabs(z_diff) > param_z * .5)
                interrupt = true;
        }
    }

    int noLines = old_trace->getNumLines();
    // the old number of lines is increased on interruption
    // for creating the new coDoLines object
    if (interrupt)
        noLines += 1;

    // creating a new coDoLines object using the number of vertices, corners
    // and lines according to old values and interruption check
    coDoLines *trace = new coDoLines(name, i + 1, i + 1, noLines);
    float *x, *y, *z;
    int *corners, *lines;
    trace->getAddresses(&x, &y, &z, &corners, &lines);

    // noLines decreased for right memcpy of old lines
    // and correct setting of new line index
    if (interrupt)
        noLines += -1;

    if (i > 0)
    {
        /* copying the values of the previous line structure into
       * the new coDoLines object */
        memcpy(x, ox, sizeof(*x) * i);
        memcpy(y, oy, sizeof(*y) * i);
        memcpy(z, oz, sizeof(*z) * i);
        memcpy(corners, ocorners, sizeof(*corners) * i);
        memcpy(lines, olines, sizeof(*lines) * noLines);
    }
    //a new corner of the line is added
    corners[i] = i;
    //a new point is added
    x[i] = x_new;
    y[i] = y_new;
    z[i] = z_new;
    // the first line starts at the beginning of the
    // corner array
    lines[0] = 0;

    // the additional line starts at the current corner index
    if (interrupt)
        lines[noLines] = i;

    return trace;
}

coDistributedObject *ComputeTrace::extractStaticData(const std::string &name, const coDoSet *set)
{
    int numParticles = m_particleSelection.size();
    coDoAbstractData *out = NULL;
    for (int t = 0; t <= m_timestepStop; ++t)
    {
        const coDoAbstractData *data = dynamic_cast<const coDoAbstractData *>(set->getElement(t));
        if (!data)
            continue;
        if (!out)
            out = data->cloneType(name, numParticles * (m_timestepStop + 1));
        for (int i = 0; i < numParticles; ++i)
        {
            out->cloneValue(t + i * (m_timestepStop + 1), data, m_particleSelection[i]);
        }
    }

    return out;
}

coDistributedObject *ComputeTrace::extractDataForParticle(const std::string &name, const coDoSet *set, int particle)
{
    coDoAbstractData *out = NULL;
    for (int t = 0; t <= m_timestepStop; ++t)
    {
        const coDoAbstractData *data = dynamic_cast<const coDoAbstractData *>(set->getElement(t));
        if (!data)
            continue;
        if (!out)
            out = data->cloneType(name, m_timestepStop + 1);
        out->cloneValue(t, data, particle);
    }

    return out;
}

coDistributedObject *ComputeTrace::extractData(const std::string &name, const coDoSet *set, bool animate)
{
    if (!animate)
        return extractStaticData(name, set);

    std::string setNameBase = name + "_1";
    std::vector<coDistributedObject *> elems;
    for (int i = 0; i < m_particleSelection.size(); ++i)
    {
        std::stringstream ss;
        ss << setNameBase << "_" << i;
        elems.push_back(extractDataForParticle(ss.str(), set, m_particleSelection[i]));
    }
    coDoSet *out = new coDoSet(setNameBase, elems.size(), &elems[0]);
    if (!animate)
        return out;

    elems.clear();
    for (int t = 0; t <= m_timestepStop; ++t)
    {
        elems.push_back(out);
    }
    return new coDoSet(name, elems.size(), &elems[0]);
}

MODULE_MAIN(Filter, ComputeTrace)
