/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SortPoints.h"
#include <do/coDoPoints.h>
#include <do/coDoData.h>

/// Constructor
SortPoints::SortPoints(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Sort points according to associated data")
{
    // Create ports:
    m_pointIn = addInputPort("points", "Points", "point data");
    m_dataIn = addInputPort("data", "Int|Float", "associated data");

    m_pointLowerOut = addOutputPort("pointsLower", "Points", "points below lower bound");
    m_dataLowerOut = addOutputPort("dataLower", "Int|Float", "data below lower bound");
    m_pointMiddleOut = addOutputPort("pointsMiddle", "Points", "points between bounds");
    m_dataMiddleOut = addOutputPort("dataMiddle", "Int|Float", "data between bounds");
    m_pointUpperOut = addOutputPort("pointsUpper", "Points", "points above uppper bounds");
    m_dataUpperOut = addOutputPort("dataUpper", "Int|Float", "data above uppper bounds");

    m_lowerBound = addFloatParam("lowerBound", "lower bound");
    m_upperBound = addFloatParam("upperBound", "upper bound");
}

template <class T>
void fillBin(T *dest, const T *source, const std::vector<int> &ind)
{
    for (int i = 0; i < ind.size(); ++i)
    {
        dest[i] = source[ind[i]];
    }
}

/// Compute routine: load checkpoint file
int SortPoints::compute(const char *)
{
    const coDistributedObject *pointsIn = m_pointIn->getCurrentObject();
    if (!pointsIn)
    {
        sendError("no point input data");
        return STOP_PIPELINE;
    }
    const coDoPoints *points = dynamic_cast<const coDoPoints *>(pointsIn);
    if (!points)
    {
        sendError("point input data not of type Points");
        return STOP_PIPELINE;
    }

    const coDistributedObject *data = m_dataIn->getCurrentObject();
    if (!data)
    {
        sendError("no data associated to points");
        return STOP_PIPELINE;
    }
    const coDoInt *intData = dynamic_cast<const coDoInt *>(data);
    const coDoFloat *floatData = dynamic_cast<const coDoFloat *>(data);
    if (!intData && !floatData)
    {
        sendError("associated data not of type Float or Int");
        return STOP_PIPELINE;
    }

    bool outputData = m_dataLowerOut->isConnected()
                      || m_dataMiddleOut->isConnected()
                      || m_dataUpperOut->isConnected();

    if (floatData && points->getNumPoints() > floatData->getNumPoints())
    {
        sendError("incompatible input data sizes");
        return STOP_PIPELINE;
    }
    if (intData && points->getNumPoints() > intData->getNumPoints())
    {
        sendError("incompatible input data sizes");
        return STOP_PIPELINE;
    }

    std::vector<int> lowerBin, middleBin, upperBin;
    if (intData)
    {
        int *ints = intData->getAddress();
        int lower = static_cast<int>(m_lowerBound->getValue());
        int upper = static_cast<int>(m_upperBound->getValue());
        for (int i = 0; i < points->getNumPoints(); ++i)
        {
            if (ints[i] < lower)
                lowerBin.push_back(i);
            else if (ints[i] > upper)
                upperBin.push_back(i);
            else
                middleBin.push_back(i);
        }
    }
    else
    {
        float *floats;
        floatData->getAddress(&floats);
        float lower = m_lowerBound->getValue();
        float upper = m_upperBound->getValue();
        for (int i = 0; i < points->getNumPoints(); ++i)
        {
            if (floats[i] < lower)
                lowerBin.push_back(i);
            else if (floats[i] > upper)
                upperBin.push_back(i);
            else
                middleBin.push_back(i);
        }
    }

    float *sx, *sy, *sz;
    points->getAddresses(&sx, &sy, &sz);
    float *x, *y, *z;
    coDoPoints *pointsLower = new coDoPoints(m_pointLowerOut->getNewObjectInfo(), (int)lowerBin.size());
    pointsLower->getAddresses(&x, &y, &z);
    fillBin(x, sx, lowerBin);
    fillBin(y, sy, lowerBin);
    fillBin(z, sz, lowerBin);
    m_pointLowerOut->setCurrentObject(pointsLower);

    coDoPoints *pointsMiddle = new coDoPoints(m_pointMiddleOut->getNewObjectInfo(), (int)middleBin.size());
    pointsMiddle->getAddresses(&x, &y, &z);
    fillBin(x, sx, middleBin);
    fillBin(y, sy, middleBin);
    fillBin(z, sz, middleBin);
    m_pointMiddleOut->setCurrentObject(pointsMiddle);

    coDoPoints *pointsUpper = new coDoPoints(m_pointUpperOut->getNewObjectInfo(), (int)upperBin.size());
    pointsUpper->getAddresses(&x, &y, &z);
    fillBin(x, sx, upperBin);
    fillBin(y, sy, upperBin);
    fillBin(z, sz, upperBin);
    m_pointUpperOut->setCurrentObject(pointsUpper);

    if (outputData)
    {
        if (intData)
        {
            int *sd = intData->getAddress();

            coDoInt *lower = new coDoInt(m_dataLowerOut->getNewObjectInfo(), (int)lowerBin.size());
            int *d = lower->getAddress();
            fillBin(d, sd, lowerBin);
            m_dataLowerOut->setCurrentObject(lower);

            coDoInt *middle = new coDoInt(m_dataMiddleOut->getNewObjectInfo(), (int)middleBin.size());
            d = middle->getAddress();
            fillBin(d, sd, middleBin);
            m_dataMiddleOut->setCurrentObject(middle);

            coDoInt *upper = new coDoInt(m_dataUpperOut->getNewObjectInfo(), (int)upperBin.size());
            d = upper->getAddress();
            fillBin(d, sd, upperBin);
            m_dataUpperOut->setCurrentObject(upper);
        }
        else
        {
            float *sd;
            floatData->getAddress(&sd);

            coDoFloat *lower = new coDoFloat(m_dataLowerOut->getNewObjectInfo(), (int)lowerBin.size());
            float *d;
            lower->getAddress(&d);
            fillBin(d, sd, lowerBin);
            m_dataLowerOut->setCurrentObject(lower);

            coDoFloat *middle = new coDoFloat(m_dataMiddleOut->getNewObjectInfo(), (int)middleBin.size());
            middle->getAddress(&d);
            fillBin(d, sd, middleBin);
            m_dataMiddleOut->setCurrentObject(middle);

            coDoFloat *upper = new coDoFloat(m_dataUpperOut->getNewObjectInfo(), (int)upperBin.size());
            upper->getAddress(&d);
            fillBin(d, sd, upperBin);
            m_dataUpperOut->setCurrentObject(upper);
        }
    }

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Filter, SortPoints)
