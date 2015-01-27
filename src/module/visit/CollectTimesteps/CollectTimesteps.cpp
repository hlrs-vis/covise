/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CollectTimesteps.h"
#include <do/coDoData.h>
#include <do/coDoSet.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <climits>

using namespace std;

#define VERBOSE

#undef DEBUG_FILE

#ifdef DEBUG_FILE
static const char *debug_filename()
{
    static char buf[64];
    sprintf(buf, "StepData.%d", getpid());
    return buf;
}

static FILE *debug = fopen(debug_filename(), "w");
#endif

/// ----- Prevent auto-generated functions by assert -------

/// Copy-Constructor: NOT IMPLEMENTED
StepData::StepData(const StepData &)
    : coModule(0, 0, "Collect single timesteps of multiple data streams")
{
    assert(0);
}

/// Assignment operator: NOT  IMPLEMENTED
StepData &StepData::operator=(const StepData &)
{
    assert(0);
    return *this;
}

/// ----- Never forget the Destructor !! -------

StepData::~StepData()
{
    // would never be called
}

inline char *STRDUP(const char *old)
{
    return strcpy(new char[strlen(old) + 1], old);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor: Main Module set-up
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

StepData::StepData(int argc, char *argv[])
    : coModule(argc, argv, "Collect single timesteps of multiple data streams")
{
    // add the startover button
    p_startOver = addBooleanParam("startOver", "Start new timelines");
    p_startOver->setValue(0);

    // add the numSteps field
    p_numSteps = addInt32Param("numSteps", "Number of steps");
    p_numSteps->setValue(0); // infinit

    // add fields for the three labels
    p_title = addStringParam("title", "Main Title");
    p_title->setValue("Step diagram");

    p_xAxis = addStringParam("xAxis", "X-Axis Title");
    p_xAxis->setValue("Step");

    p_yAxis = addStringParam("yAxis", "Y-Axis Title");
    p_yAxis->setValue("Value");

    // minimum and maximum, set to same for auto-scale
    p_min = addFloatParam("min", "Minimum value for Y axis");
    p_min->setValue(0.0);

    p_max = addFloatParam("max", "Maximum value for Y axis");
    p_max->setValue(0.0);

    // the choices
    static const char *select[] = { " --- " };

    char buffer[16];
    int i;
    for (i = 0; i < MAX_LINES; i++)
    {
        sprintf(buffer, "Line_%d", i);
        p_selectLine[i] = addChoiceParam(buffer, "Select value for line");
        p_selectLine[i]->setValue(1, select, 0);
    }

    // and the input port for the residuals
    p_in = addInputPort("inPort", "StepData", "Step data input");

    // output port for plot data
    p_out = addOutputPort("out_port", "Vec2",
                          "set of two variables");

    // Dummy element at the start
    d_data = new DataRec;
    d_data->data = NULL;
    d_data->next = NULL;
    d_data->realtime = 0.0;

    // don't know the number of fields right now
    d_numFields = 0;

    // No steps yet
    d_numSteps = 0;

    // Last Object empty
    objName[0] = '\0';

    // No labels yet
    d_label = NULL;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++  param callback: called by immediate-mode parameters
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void StepData::param(const char *paraName, bool)
{
    // the 'starover' button was pressed
    if (strcmp(paraName, p_startOver->getName()) == 0)
    {
        sendInfo("Starting new lines");

        // unpress button
        p_startOver->setValue(0);

        // discard saved data;
        clearDataBase();
    }
    else if (strcmp(paraName, p_numSteps->getName()) == 0)
    {
        int numSteps = p_numSteps->getValue();
        if (numSteps < 0 || numSteps == 1)
        {
            sendInfo("corrected illegal numSteps stting");
            p_numSteps->setValue(0);
        }
        while (numSteps < d_numSteps) // reducing #steps
        {
            DataRec *killElem = d_data->next;
            d_data->next = killElem->next;
            delete[] killElem -> data;
            delete killElem;
            d_numSteps--;
        }
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++  update choice labels
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void StepData::clearDataBase()
{
    // delete old data chain
    DataRec *next, *curr;
    curr = d_data->next; // keep dummy at start
    while (curr)
    {
        next = curr->next;
        delete[] curr -> data;
        delete curr;
        curr = next;
    }
    d_last = d_data;
    d_numSteps = 0;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++  update choice labels
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int StepData::updateLabels(const coDoFloat *data, int numElem)
{
    const char **name, **val;
    int i;
    int changeLabels = 0; // check whether we could

    /// delete old labels : ok with NULL deleting
    for (i = 0; i < d_numFields; i++)
        delete[] d_label[i];
    delete[] d_label;

    d_label = new char *[numElem + 1];
    d_label[0] = STRDUP(" -- No line -- ");
    int numAttrib = data->getAllAttributes(&name, &val);
    int numLabels = 1;

    // get all LABEL attribs
    for (i = 0; i < numAttrib; i++)
        if (strcmp(name[i], "LABEL") == 0)
            d_label[numLabels++] = STRDUP(val[i]);

    // check for correct number
    if (numLabels != numElem + 1)
    {
        sendError("Illegal number of LABEL attribs");
        for (i = 0; i < numLabels; i++)
            delete[] d_label[i];
        delete[] d_label;
        d_label = NULL;
        return -1;
    }
    else
        d_numFields = numElem; // we set this only if we're ok

    // set choice labels, keep setting if valid
    for (i = 0; i < MAX_LINES; i++)
    {
        int oldVal = p_selectLine[i]->getValue();
        if (oldVal <= numLabels)
            p_selectLine[i]->setValue(numLabels, d_label, oldVal);
        else
        {
            p_selectLine[i]->setValue(numLabels, d_label, 0);
            changeLabels = 1;
        }
    }
    return changeLabels;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++  find good min/max/step for a given interval
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void StepData::properMinMax(float &min, float &max,
                            float &stepBig, float &stepSmall,
                            int numSteps)
{
    float x;
    if (max < min)
    {
        x = min;
        min = max;
        max = x;
    }
    numSteps--; // number of intervals
    if (numSteps < 0)
        numSteps = 1;

    // make sure we do have an interval.
    if (min == max)
        max = min + 1;

    stepBig = (max - min) / numSteps;

    float factList[8] = { 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0 };
    //                    -4    -3   -2   -1    0    1    2     3
    float *fact = factList + 4;

    int i = 0;
    x = 1.0;
    double tenFact = 1.0;

    // find the big stepping
    if (x >= stepBig)
        while (x >= stepBig)
        {
            i--;
            if (i % 3 == 0)
                tenFact /= 10.0;
            x = fact[i % 3] * tenFact;
        }
    else
        while (x < stepBig)
        {
            i++;
            if (i % 3 == 0)
                tenFact *= 10.0;
            x = fact[i % 3] * tenFact;
        }
    stepBig = x;

    // find the sub-stepping
    i %= 3;
    i -= 2;

    stepSmall = fact[i] * tenFact;

    // make min/max integer multiples of stepBig
    i = (int)(min / stepBig);
    x = stepBig * i;
    if (x > min)
        min = stepBig * (i - 1);
    else
        min = x;

    i = (int)(max / stepBig);
    x = stepBig * i;
    if (x < max)
        max = stepBig * (i + 1);
    else
        max = x;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++  compute callback: called when data arrives
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int StepData::compute(const char *)
{
    const coDistributedObject *inObj = p_in->getCurrentObject();
    if (!inObj->isType("USTSDT"))
    {
        sendError("Illegal data type received");
        return FAIL;
    }

    const coDoFloat *data = (const coDoFloat *)inObj;

    // if the number of elements changed, it makes no sense co continue,
    // and we have to update the parameters
    int numElem = data->getNumPoints();
    int initFlag = 0;
    if (d_numFields != numElem)
    {
        initFlag = updateLabels(data, numElem);
        clearDataBase();
    }

    //////////// we only add another step if it's not another exec of old data
    if (strcmp(data->getName(), objName))
    {
        int maxSteps = p_numSteps->getValue();

        // we need this if case no realtime Attrib or illegal value
        float oldtime = d_last->realtime;

        if (maxSteps == 0 || d_numSteps < maxSteps)
        {
            d_last->next = new DataRec; // chain new container at end
            d_last = d_last->next; // move last-pointer to new end
            d_last->next = NULL; // zero own chain pointer
            d_last->data = new float[numElem]; // alloc data entrz
            d_numSteps++; // one more step
        }
        else
        {
            DataRec *reuse = d_data->next; // re-use first container
            d_data->next = d_data->next->next; // unlink from chain head
            d_last->next = reuse; // chain in at end
            reuse->next = NULL; // zero own chain pointer
            d_last = reuse; // move last-pointer to new end
        }

        //////////// copy data to container
        float *dataArr;
        data->getAddress(&dataArr);

#ifdef DEBUG_FILE
        //////////// dump input data
        {
            int i;
            fprintf(debug, "\n------ received data: %d elem\n\n", numElem);
            for (i = 0; i < numElem; i++)
                fprintf(debug, "  %-3i : %10.5f\n", i, dataArr[i]);
        }
#endif

        memcpy(d_last->data, dataArr, numElem * sizeof(float));

        //////////// retrieve REALTIME attribute
        float newtime = 0.0;
        const char *attr = data->getAttribute("REALTIME");
        if (attr)
            newtime = atof(attr);
        if (newtime == 0.0 || newtime < oldtime)
            newtime = oldtime + 1.0;
        d_last->realtime = newtime;

#ifdef DEBUG_FILE
        //////////// dump internal list
        {
            int i;
            fprintf(debug, "\n------ List now ------\n");
            DataRec *elem = d_data->next;
            while (elem)
            {
                fprintf(debug, "- Record: realtime= %f\n", elem->realtime);
                if (elem == d_last)
                    fprintf(debug, "  | this is d_last\n");
                for (i = 0; i < d_numFields; i++)
                    fprintf(debug, "  |-%-3i : %10.5f\n", i, elem->data[i]);
                fprintf(debug, "  +--------------------\n");
                elem = elem->next;
            }
        }
#endif

        //////////// update step data name
        strcpy(objName, data->getName());
    }

    // If we've just created new choices or have no data, we won't exec here
    if (initFlag || !d_numSteps)
        return FAIL;

    //////////// create the lines

    coDistributedObject *lines[MAX_LINES];
    int numLines = 0;
    char nameBase[128], name[128];
    sprintf(nameBase, "%s_%%d", p_out->getObjName());
    char *label[MAX_LINES];

    float startTime = FLT_MAX;
    float endTime = -FLT_MAX;
    float minVal = FLT_MAX;
    float maxVal = -FLT_MAX;

    int line;
    for (line = 0; line < MAX_LINES; line++)
    {
        int select = p_selectLine[line]->getValue() - 3;
        if (select >= 0)
        {
            //////////// allocate DO
            sprintf(name, nameBase, line);
            coDoVec2 *line
                = new coDoVec2(name, d_numSteps);
            float *x, *y;
            line->getAddresses(&x, &y);

            DataRec *actStep = d_data->next; // skip dummy
            while (actStep)
            {
                *x = actStep->realtime;
                *y = actStep->data[select];

                if (*x < startTime)
                    startTime = *x;
                if (*x > endTime)
                    endTime = *x;
                if (*y < minVal)
                    minVal = *y;
                if (*y > maxVal)
                    maxVal = *y;

                x++;
                y++;
                actStep = actStep->next;
            }
            label[numLines] = d_label[select + 1];
            lines[numLines] = line;
            numLines++;
        }
    }

    // no output object created
    if (numLines == 0)
        return FAIL;

    // check user's settings for min and max - user overrides auto
    float uMin = p_min->getValue();
    float uMax = p_max->getValue();

    if (uMin != uMax) // don't check uMin<uMax - User might want to mirror diag.
    {
        minVal = uMin;
        maxVal = uMax;
    }

    // calculate some nice-looking ticks and bounds
    if (startTime == endTime)
        endTime = startTime + 1;
    if (minVal == maxVal)
        maxVal = minVal + 1;

    float tBigTick, tSmallTick, vBigTick, vSmallTick;

    properMinMax(startTime, endTime, tBigTick, tSmallTick, 6);
    properMinMax(minVal, maxVal, vBigTick, vSmallTick, 5);

    //////////// create the set
    coDoSet *set = new coDoSet(p_out->getObjName(), numLines, lines);

    //////////// Plot attributes
    char buffer[65536], add[256];
    buffer[0] = '\0';

    sprintf(add, "TITLE \"%s\"\n", p_title->getValue());
    strcat(buffer, add);
    sprintf(add, "xaxis LABEL \"%s\"\n", p_xAxis->getValue());
    strcat(buffer, add);
    sprintf(add, "yaxis LABEL \"%s\"\n", p_yAxis->getValue());
    strcat(buffer, add);

    strcat(buffer, "FRAME ON\n");
    strcat(buffer, "LEGEND ON \n");
    strcat(buffer, "LEGEND BOX ON \n");
    strcat(buffer, "LEGEND BOX FILL ON\n");
    strcat(buffer, "LEGEND BOX FILL COLOR 0\n");

    for (line = 0; line < numLines; line++)
    {
        sprintf(add, "LEGEND STRING %d \"%s\"\n", line, label[line]);
        strcat(buffer, add);
        sprintf(add, "S%d COLOR %d\n", line, line % 12 + 2);
        strcat(buffer, add);
    }

    strcat(buffer, "SETS linewidth 2\n");

    sprintf(add, "WORLD %f,%f,%f,%f\n",
            startTime, minVal, endTime, maxVal);
    strcat(buffer, add);

    sprintf(add, "YAXIS TICK MAJOR %f\nYAXIS TICK MINOR %f\n",
            vBigTick, vSmallTick);
    strcat(buffer, add);

    sprintf(add, "XAXIS TICK MAJOR %f\nXAXIS TICK MINOR %f\n",
            tBigTick, tSmallTick);
    strcat(buffer, add);

    set->addAttribute("COMMANDS", buffer);
    //cerr << "COMMANDS: \n" << buffer << endl;

    p_out->setCurrentObject(set);

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++  create the module and start it
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char *argv[])

{
    // create the module
    StepData *application = new StepData(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
