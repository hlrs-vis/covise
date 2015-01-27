/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <api/coModule.h>
using namespace covise;
class CoolEmAllSimulation : public coModule
{

private:
    virtual int compute(const char *port);
    virtual void param(const char *, bool inMapLoading);
    virtual void postInst();
    virtual void quit();

    coStringParam *debbURL;
    coStringParam *experimentID;
    coStringParam *trialID;
    coIntScalarParam *experimentType;
    coStringParam *databasePath;
    coStringParam *toSimulate;
    coStringParam *startTime;
    coStringParam *endTime;

public:
    CoolEmAllSimulation(int argc, char *argv[]);
};
