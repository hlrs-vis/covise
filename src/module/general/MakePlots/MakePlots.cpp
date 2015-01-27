/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include "MakePlots.h"

MakePlots::MakePlots(int argc, char *argv[])
    : coModule(argc, argv, "Collect scalar objects in objects for Plot")
{
    p_title = addStringParam("title", "Main Title");
    p_title->setValue("Step diagram");

    p_xAxis = addStringParam("xAxis", "X-Axis Title");
    p_xAxis->setValue("Step");

    p_yAxis = addStringParam("yAxis", "Y-Axis Title");
    p_yAxis->setValue("Value");

    p_auto = addBooleanParam("auto", "auto or manual");
    p_auto->setValue(1);

    float Inilimits[2] = // use default, if we have these values
        {
          0.0, 0.0
        };
    p_userXlimits = addFloatVectorParam("Xlimits", "Min and max for X-axis");
    p_userXlimits->setValue(2, Inilimits);
    p_userYlimits = addFloatVectorParam("Ylimits", "Min and max for Y-axis");
    p_userYlimits->setValue(2, Inilimits);

    long IniXticks[2] = { TBIGTICK, TSMALLTICK };
    long IniYticks[2] = { VBIGTICK, VSMALLTICK };
    p_userXticks = addInt32VectorParam("Xticks", "Major and minor tick for X-axis");
    p_userXticks->setValue(2, IniXticks);
    p_userYticks = addInt32VectorParam("Yticks", "Major and minor tick for Y-axis");
    p_userYticks->setValue(2, IniYticks);

    // ports
    p_x = addInputPort("inPort_x", "Float", "X magnitude");
    p_x->setRequired(1);
    int i;
    const char *name = "inPort_y";
    const char *desc = "Y magnitude";
    char full_name[64];
    char full_desc[64];
    char tail[16];
    for (i = 0; i < NO_MAX_PLOTS; ++i)
    {
        sprintf(tail, "_%d", i);
        strcpy(full_name, name);
        strcat(full_name, tail);
        strcpy(full_desc, desc);
        strcat(full_desc, tail);
        p_y[i] = addInputPort(full_name, "Float", full_desc);
        p_y[i]->setRequired(0);
    }
    p_out = addOutputPort("outPlots", "Vec2", "Plots");
}

void MakePlots::param(const char *paramname, bool /*inMapLoading*/)
{
    if (strcmp(p_auto->getName(), paramname) == 0)
    {
        if (p_auto->getValue())
        {
            p_userXlimits->disable();
            p_userYlimits->disable();
            p_userXticks->disable();
            p_userYticks->disable();
        }
        else
        {
            p_userXlimits->enable();
            p_userYlimits->enable();
            p_userXticks->enable();
            p_userYticks->enable();
        }
    }
}

int MakePlots::Diagnose()
{
    // check manual ticking
    if (!p_auto->getValue() && (!p_userXticks->getValue(0) || !p_userXticks->getValue(1) || !p_userYticks->getValue(0) || !p_userYticks->getValue(1)))
    {
        sendError("Number of ticks may only be positive");
        return -1;
    }
    if (!p_auto->getValue() && p_userXticks->getValue(0) > p_userXticks->getValue(1))
    {
        sendError("Number of X big ticks may not outnumber that of the small ones");
        return -1;
    }
    if (!p_auto->getValue() && p_userXlimits->getValue(0) >= p_userXlimits->getValue(1))
    {
        sendError("The X lower limit may not be greater than nor equal to the upper one");
        return -1;
    }
    if (!p_auto->getValue() && p_userYticks->getValue(0) > p_userYticks->getValue(1))
    {
        sendError("Number of Y big ticks may not outnumber that of the small ones");
        return -1;
    }
    if (!p_auto->getValue() && p_userYlimits->getValue(0) >= p_userYlimits->getValue(1))
    {
        sendError("The Y lower limit may not be greater than nor equal to the upper one");
        return -1;
    }
    // check X-data
    if (!p_x->getCurrentObject())
    {
        sendError("X-object is not available");
        return -1;
    }
    if (!(p_x->getCurrentObject()->objectOk()))
    {
        sendError("X-object is not OK");
        return -1;
    }
    if (!(p_x->getCurrentObject()->isType("USTSDT")))
    {
        sendError("Input object type should be USTSDT");
        return -1;
    }
    noPoints_ = ((coDoFloat *)(p_x->getCurrentObject()))->getNumPoints();

    // check Y-data
    int i, j;
    for (i = 0, j = 0; i < NO_MAX_PLOTS; ++i)
    {
        if (!p_y[i]->getCurrentObject())
        {
            continue;
        }
        ++j;
        if (!(p_y[i]->getCurrentObject()->objectOk()))
        {
            sendError("A Y-object is not OK");
            return -1;
        }
        if (!(p_y[i]->getCurrentObject()->isType("USTSDT")))
        {
            sendError("Input object type should be USTSDT");
            return -1;
        }
    }
    if (j == 0)
    {
        sendError("At least one Y-object is necessary");
        return -1;
    }
    noPlots_ = j;
    // check lengths
    for (i = 0, j = 0; i < NO_MAX_PLOTS; ++i)
    {
        if (!p_y[i]->getCurrentObject())
        {
            continue;
        }
        if (((coDoFloat *)(p_y[i]->getCurrentObject()))->getNumPoints() != noPoints_)
        {
            sendError("X- and Y-objects should have the same length");
            return -1;
        }
    }
    // now set variables for plot scaling
    if (!p_auto->getValue())
    {
        startTime_ = p_userXlimits->getValue(0);
        endTime_ = p_userXlimits->getValue(1);
    }
    else
    {
        startTime_ = FLT_MAX;
        endTime_ = -FLT_MAX;
    }
    if (!p_auto->getValue())
    {
        minVal_ = p_userYlimits->getValue(0);
        maxVal_ = p_userYlimits->getValue(1);
    }
    else
    {
        minVal_ = FLT_MAX;
        maxVal_ = -FLT_MAX;
    }

    return 0;
}

int MakePlots::compute(const char *)
{
    if (Diagnose() < 0)
        return FAIL;
    int i, j;
    coDistributedObject **outplots = new coDistributedObject *[noPlots_ + 1];
    outplots[noPlots_] = 0;
    float *xData;
    float *yData;
    ((coDoFloat *)(p_x->getCurrentObject()))->getAddress(&xData);

    // x-limits
    if (p_auto->getValue())
    {
        for (i = 0; i < noPoints_; ++i)
        {
            if (xData[i] < startTime_)
                startTime_ = xData[i];
            if (xData[i] > endTime_)
                endTime_ = xData[i];
        }
    }
    else if (startTime_ == endTime_)
    {
        endTime_ = startTime_ + 1.0f;
    }

    char short_buffer[16];
    int k;
    for (i = 0, j = 0; i < NO_MAX_PLOTS; ++i)
    {
        if (!p_y[i]->getCurrentObject())
            continue;
        std::string outname(p_out->getObjName());
        sprintf(short_buffer, "_%d", j);
        outname += short_buffer;
        ((coDoFloat *)(p_y[i]->getCurrentObject()))->getAddress(&yData);
        outplots[j] = new coDoVec2(outname, noPoints_, xData, yData);
        // y limits
        if (p_auto->getValue())
        {
            for (k = 0; k < noPoints_; ++k)
            {
                if (yData[k] < minVal_)
                    minVal_ = yData[k];
                if (yData[k] > maxVal_)
                    maxVal_ = yData[k];
            }
        }

        ++j;
    }
    if (!p_auto->getValue() && minVal_ == maxVal_)
    {
        maxVal_ = minVal_ + 1.0f;
    }

    if (p_auto->getValue())
    {
        properMinMax(startTime_, endTime_, tBigTick_, tSmallTick_, TBIGTICK);
        properMinMax(minVal_, maxVal_, vBigTick_, vSmallTick_, VBIGTICK);
    }
    else
    {
        tBigTick_ = (endTime_ - startTime_) / p_userXticks->getValue(0);
        tSmallTick_ = (endTime_ - startTime_) / p_userXticks->getValue(1);
        vBigTick_ = (maxVal_ - minVal_) / p_userYticks->getValue(0);
        vSmallTick_ = (maxVal_ - minVal_) / p_userYticks->getValue(1);
    }

    coDoSet *outset = new coDoSet(p_out->getObjName(), outplots);
    // COMMAND attribute
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

    for (i = 0, j = 0; i < NO_MAX_PLOTS; i++) // j is the color index
    {
        if (!p_y[i]->getCurrentObject())
            continue;
        sprintf(add, "LEGEND STRING %d \"%s\"\n", j,
                p_y[i]->getCurrentObject()->getAttribute("SPECIES"));
        strcat(buffer, add);
        sprintf(add, "S%d COLOR %d\n", j, j % 12 + 2);
        strcat(buffer, add);
        ++j;
    }

    strcat(buffer, "SETS linewidth 2\n");

    sprintf(add, "WORLD %f,%f,%f,%f\n",
            startTime_, minVal_, endTime_, maxVal_);
    strcat(buffer, add);

    sprintf(add, "YAXIS TICK MAJOR %f\nYAXIS TICK MINOR %f\n",
            vBigTick_, vSmallTick_);
    strcat(buffer, add);

    sprintf(add, "XAXIS TICK MAJOR %f\nXAXIS TICK MINOR %f\n",
            tBigTick_, tSmallTick_);
    strcat(buffer, add);

    outset->addAttribute("COMMANDS", buffer);
    /////////////////////////

    for (i = 0; i < noPlots_; ++i)
    {
        delete outplots[i];
    }
    delete[] outplots;
    p_out->setCurrentObject(outset);
    return SUCCESS;
}

void MakePlots::properMinMax(float &min, float &max,
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

    float factList[8] = { 0.05f, 0.1f, 0.2f, 0.5f, 1.0f, 2.0f, 5.0f, 10.0f };
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
            x = (float)(fact[i % 3] * tenFact);
        }
    else
        while (x < stepBig)
        {
            i++;
            if (i % 3 == 0)
                tenFact *= 10.0;
            x = (float)(fact[i % 3] * tenFact);
        }
    stepBig = x;
    // find the sub-stepping
    i %= 3;
    i -= 2;

    stepSmall = (float)(fact[i] * tenFact);

    // make min/max integer multiples of stepBig
    i = int(min / stepBig);
    x = float(stepBig * i);
    if (x > min)
        min = stepBig * (i - 1);
    else
        min = x;

    i = int(max / stepBig);
    x = float(stepBig * i);
    if (x < max)
        max = stepBig * (i + 1);
    else
        max = x;
}

MODULE_MAIN(Tools, MakePlots)
