/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file PeriodicRotResults.h
 * completes results to a full 360° rotation.
 * rotation axís has to be z-axis.
 */

// #include "PeriodicRotResults.h"  // completes results/meshes to a full 360° rotation.

#ifndef __PeriodicRotResults_h__
#define __PeriodicRotResults_h__

#include "ResultsFileData.h" // a container for results file data.
#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include "os.h" // operating system dependent definitions
#include <do/coDoSet.h>
#include <do/coDoData.h>

/**
 * completes results to a full 360° rotation.
 * rotation axís has to be z-axis.
 */
class PeriodicRotResults : public ResultsFileData
{
public:
    PeriodicRotResults(double symmAngleDeg,
                       int noOfStepsPerBlock,
                       ResultsFileData *originalData,
                       OutputHandler *outputHandler);
    virtual ~PeriodicRotResults(){};

    // inherited getter methods
    virtual int getNoOfTimeSteps(void) const;
    virtual int getNoOfDataTypes(int timeStepNo) const;
    virtual std::string getDataTypeName(int timeStepNo, int dataTypeNo);
    virtual bool getDisplacements(int timeStepNo, float *dx[], float *dy[], float *dz[]);
    virtual ResultsFileData::EDimension getDimension(int timeStepNo, int dataTypeNo);
    virtual coDistributedObject *getDataObject(int dataTypeNo, int timeStepNo);

private:
    OutputHandler *_outputHandler;

    // original data
    int _noOfStepsPerBlock;
    double _symmAngle;
    ResultsFileData *_originalData;

    // derived data
    int _noOfBlocks;
    int _noOfTimeSteps;

    coDoFloat *getDataObjectCopy(
        coDoFloat *scalarData,
        std::string dataName) const;

    coDoVec3 *rotateDataObject(
        coDoVec3 *vectorData,
        std::string dataName,
        double angle) const;

    PointCC rotateVector(const PointCC &v,
                         double angle) const;
};

#endif
