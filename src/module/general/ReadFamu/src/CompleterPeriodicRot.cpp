/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file CompleterPeriodicRot.h
 * completes results to a full 360° rotation.
 * rotation axís has to be z-axis.
 */

#include "CompleterPeriodicRot.h" // completes results/meshes to a full 360° rotation.
#include "errorinfo.h" // a container for error data.
#include "coordconv.hxx" // conversion between coordinate systems.

#define PI 3.14159265359

CompleterPeriodicRot::CompleterPeriodicRot(double symmAngleDeg,
                                           int noOfStepsPerBlock,
                                           OutputHandler *outputHandler)
    : _outputHandler(outputHandler)
    , _noOfStepsPerBlock(noOfStepsPerBlock)
    , _symmAngle((symmAngleDeg / 360) * 2 * PI)
    , _noOfBlocks(0)
    , _noOfTimeSteps(0)
{
    ASSERT(symmAngleDeg <= 180, _outputHandler)
    ASSERT(_noOfStepsPerBlock > 0, _outputHandler)
    ASSERT(fabs(((int)(360 / symmAngleDeg) - (360 / symmAngleDeg))) < 1e-6, _outputHandler);

    _noOfBlocks = (int)(PI / _symmAngle);
    _noOfTimeSteps = _noOfBlocks * _noOfStepsPerBlock;
}

coDistributedObject *CompleterPeriodicRot::getDataObject(int dataTypeNo, int timeStepNo)
{
    coDistributedObject *retval = NULL;
    if (timeStepNo < _noOfStepsPerBlock)
    {
        // return original data
        retval = _originalData->getDataObject(dataTypeNo, timeStepNo);
    }
    else
    {
        // get original data of respective data
        coDistributedObject *origData = _originalData->getDataObject(dataTypeNo, timeStepNo % _noOfTimeSteps);
        std::string origName = _originalData->getDataTypeName(timeStepNo % _noOfTimeSteps, dataTypeNo);
        ResultsFileData::EDimension dim = _originalData->getDimension(timeStepNo % _noOfTimeSteps, dataTypeNo);

        if (dim == ResultsFileData::SCALAR)
        {
            // scalar data stay unchanged; just copy the original data
            retval = getDataObjectCopy((coDoFloat *)origData, origName);
        }
        else
        {
            // rotate vector data
        }
    }
    return retval;
}

coDoFloat *CompleterPeriodicRot::getDataObjectCopy(
    coDoFloat *scalarData,
    std::string dataName) const
{
    const int noOfPoints = scalarData->getNumPoints();
    float *origValues = NULL;
    scalarData->getAddress(&origValues);

    // copy data
    float *copyValues = new float[noOfPoints];
    INT i;
    for (i = 0; i < noOfPoints; i++)
    {
        copyValues[i] = origValues[i];
    }
    coDoFloat *retval = new coDoFloat(dataName.c_str(), noOfPoints, copyValues);
    delete[] copyValues;
    return retval;
}

coDoVec3 *CompleterPeriodicRot::rotateDataObject(
    coDoVec3 *vectorData,
    std::string dataName,
    double angle) const
{
    // get original data
    const int noOfPoints = vectorData->getNumPoints();
    float *vx_orig = NULL;
    float *vy_orig = NULL;
    float *vz_orig = NULL;
    vectorData->getAddresses(&vx_orig, &vy_orig, &vz_orig);

    // copy & rotate data
    float *vx_cp = new float[noOfPoints];
    float *vy_cp = new float[noOfPoints];
    float *vz_cp = new float[noOfPoints];
    INT i;
    for (i = 0; i < noOfPoints; i++)
    {
        PointCC v = { vx_orig[i], vy_orig[i], vz_orig[i] };
        v = rotateVector(v, angle);
        vx_cp[i] = float(v.x);
        vy_cp[i] = float(v.y);
        vz_cp[i] = float(v.z);
    }
    coDoVec3 *retval = new coDoVec3(dataName.c_str(), noOfPoints, vx_cp, vy_cp, vz_cp);
    delete[] vx_cp;
    delete[] vy_cp;
    delete[] vz_cp;
    return retval;
}

PointCC CompleterPeriodicRot::rotateVector(const PointCC &v,
                                           double angle) const
{
    double rho, phi, z;
    coordConv::kart2zy(v.x, v.y, v.z, rho, phi, z);

    PointCC retval = v;
    phi += angle;
    coordConv::zy2kart(rho, phi, z, retval.x, retval.y, retval.z);
    return retval;
}

// --------------------------------- trivial getter methods ---------------------------------

int CompleterPeriodicRot::getNoOfTimeSteps(void) const
{
    return _noOfTimeSteps;
}

int CompleterPeriodicRot::getNoOfDataTypes(int timeStepNo) const
{
    return _originalData->getNoOfDataTypes(timeStepNo % _noOfTimeSteps);
}

std::string CompleterPeriodicRot::getDataTypeName(int timeStepNo, int dataTypeNo)
{
    return _originalData->getDataTypeName(timeStepNo % _noOfTimeSteps, dataTypeNo);
}

bool CompleterPeriodicRot::getDisplacements(int timeStepNo,
                                            float *dx[], float *dy[], float *dz[])
{
    return _originalData->getDisplacements(timeStepNo % _noOfTimeSteps, dx, dy, dz);
}
