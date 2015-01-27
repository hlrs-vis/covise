/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __STEP_DATA_H_
#define __STEP_DATA_H_

// 28.03.00

#include <api/coModule.h>
using namespace covise;
#include <do/coDoData.h>

/**
 * Andreas Werner 04/2000
 *
 * StepData Module collects single timesteps of multiple data streams
 * and creates plot input
 */
class StepData : public coModule
{

private:
    enum
    {
        MAX_LINES = 10
    };

    /// React on incoming data stream
    virtual int compute(const char *);

    /// React on parameter changes
    virtual void param(const char *, bool);

    /// Copy-Constructor: NOT  IMPLEMENTED
    StepData(const StepData &);

    /// Assignment operator: NOT  IMPLEMENTED
    StepData &operator=(const StepData &);

    // update Choice labels by LABEL attribs of incoming objects
    int updateLabels(const coDoFloat *data, int numElem);

    // calc proper-looking min,max,step from min and max input
    void properMinMax(float &min, float &max,
                      float &stepBig, float &stepSmall,
                      int numSteps);

    // clear database
    void clearDataBase();

    // Parameters
    coBooleanParam *p_startOver;
    coIntScalarParam *p_numSteps;
    coFloatParam *p_min, *p_max;
    coChoiceParam *p_selectLine[MAX_LINES];
    coStringParam *p_title, *p_xAxis, *p_yAxis;

    // Ports
    coInputPort *p_in;
    coOutputPort *p_out;

    // data field records
    struct DataRec
    {
        DataRec *next;
        float *data;
        float realtime;
    } *d_data, *d_last;

    // number of different values in each step
    int d_numFields;

    // number of steps currently in queue
    int d_numSteps;

    // the name of the last object we received
    char objName[128];

    // the labels on the choices are also in the legend
    char **d_label;

public:
    /// Create the module
    StepData(int argc, char *argv[]);

    /// Destructor : virtual in case we derive objects
    virtual ~StepData();
};
#endif
