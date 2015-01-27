/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "timesteps.h"

using namespace DTF;

CLASSINFO_OBJ(ClassInfo_DTFTimeSteps, TimeSteps, "DTF::TimeSteps", 1);

TimeSteps::TimeSteps()
    : Tools::BaseObject()
{
    timeSteps.clear();
    INC_OBJ_COUNT(getClassName());
}

TimeSteps::TimeSteps(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    timeSteps.clear();
    INC_OBJ_COUNT(getClassName());
}

TimeSteps::~TimeSteps()
{
    timeSteps.clear();
    DEC_OBJ_COUNT(getClassName());
}

bool TimeSteps::setData(vector<Data *> data)
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    TimeStep *timeStep = NULL;
    Data *tsData = NULL;

    clear();

    if (data.size() != 0)
    {
        for (int i = 0; i < data.size(); i++)
        {
            timeStep = (TimeStep *)cm->getObject("DTF::TimeStep");

            tsData = data[i];

            if (tsData != NULL)
                if (fillTimeStep(timeStep, tsData))
                    timeSteps.push_back(timeStep);

            timeStep = NULL;
            tsData != NULL;
        }

        return true;
    }

    return false;
}

bool TimeSteps::fillTimeStep(TimeStep *timeStep, Data *data)
{
    int numSims = 0;
    int simNum = 1;
    int numZones = 0;
    Sim *sim = NULL;

    if ((data != NULL) && (timeStep != NULL))
        if ((numSims = data->getNumSims()) >= 1)
            if ((sim = data->getSim(simNum)) != NULL)
                if (timeStep->setData(sim, 0))
                    return true;

    return false;
}

TimeStep *TimeSteps::get(int timeStepNum)
{
    if (timeStepNum < timeSteps.size())
        return timeSteps[timeStepNum];

    return NULL;
}

int TimeSteps::getNumTimeSteps()
{
    return timeSteps.size();
}

void TimeSteps::clear()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    vector<TimeStep *>::iterator tsIterator = timeSteps.begin();

    while (tsIterator != timeSteps.end())
    {
        TimeStep *timeStep = *tsIterator;

        if (timeStep != NULL)
        {
            timeStep->clear();
            cm->deleteObject(timeStep->getID());

            timeStep = NULL;
        }

        ++tsIterator;
    }

    timeSteps.clear();
}

void TimeSteps::print()
{
    cout << "contained data in timesteps:" << endl;
    cout << "----------------------------" << endl;

    vector<TimeStep *>::iterator tsIterator = timeSteps.begin();
    int i = 0;

    while (tsIterator != timeSteps.end())
    {
        TimeStep *timeStep = *tsIterator;

        if (timeStep != NULL)
        {
            cout << "timeStep #" << ++i << ": " << endl;
            timeStep->print();
            cout << endl;
        }

        ++tsIterator;
    }
}
