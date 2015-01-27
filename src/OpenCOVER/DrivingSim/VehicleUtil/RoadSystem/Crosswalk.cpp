/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Crosswalk.h"
#include <stdio.h>

//using namespace covise;
//using namespace opencover;

const int Crosswalk::DONOTENTER = -2;
const int Crosswalk::EXITERROR = -4;

/**
 * Create a new crosswalk object
 */
Crosswalk::Crosswalk(const std::string &id_, const std::string &name_, const double s_, const double length_, const double crossProb_, const double resetTime_, const std::string &type_, int dbg_)
    : Element(id_)
    , lastActivityTime(0.0)
    , id(id_)
    , name(name_)
    , s(s_)
    , length(length_)
    , crossProb(crossProb_)
    , resetTime(resetTime_)
    , type(type_)
    , validFromLane(INT_MIN)
    , validToLane(INT_MAX)
    , curPedId(0)
    , curWaitId(0)
    , curVehId(0)
    , debugLvl(dbg_)
{
    if (debugLvl > 0)
        printDebug(1, "constructed at pos=%.2f with length=%.2f; crossProb=%.2f; resetTime=%.2f", s, length, crossProb, resetTime);
}

/**
 * Define the lanes for which the crosswalk exists
 */
void Crosswalk::setValidity(const int from, const int to)
{
    validFromLane = from;
    validToLane = to;

    // Test that (fromLane < toLane)
    if (from > to)
    {
        int tmp = validFromLane;
        validFromLane = validToLane;
        validToLane = tmp;
    }
}
std::pair<int, int> Crosswalk::getValidity() const
{
    return std::pair<int, int>(validFromLane, validToLane);
}

/**
 * Attempt to let a new pedestrian enter the crosswalk
 * If no cars are currently driving through, assign the pedestrian an ID
 * Otherwise, return DONOTENTER
 */
int Crosswalk::enterPedestrian(const double time)
{
    // If there are no vehicles in the crosswalk, assign the pedestrian a unique id
    if (vehIdsVector.empty())
    {
        // Pedestrian will be able to enter, so update timer
        lastActivity(time);

        if (pedIdsVector.empty())
            curPedId = 0; // reset ids if this is the only pedestrian

        // Increment and assign
        curPedId++;
        pedIdsVector.push_back(curPedId);
        if (debugLvl > 0)
            printDebug(1, "pedestrian# %d allowed to enter crosswalk, %d currently crossing", curPedId, pedIdsVector.size());
        return curPedId;
    }

    // If the crosswalk is unsafe, do not allow pedestrian to enter
    else
    {
        if (debugLvl > 0)
            printDebug(3, "pedestrian not allowed to enter, %d vehicles to wait for, %d pedestrians waiting already", vehIdsVector.size(), waitIdsVector.size());
        return DONOTENTER;
    }
}
/**
 * Pedestrian with the given ID is done crossing, so remove his ID from the list
 */
int Crosswalk::exitPedestrian(const int pedId, const double time)
{
    if (pedId < 0)
        return EXITERROR;

    // Pedestrian exiting, so update timer
    lastActivity(time);

    // Find the pedestrian's id in the vector, remove it, and return it
    for (std::vector<int>::iterator idIt = pedIdsVector.begin(); idIt != pedIdsVector.end(); idIt++)
    {
        if (pedId == (*idIt))
        {
            pedIdsVector.erase(idIt);
            if (debugLvl > 0)
                printDebug(1, "pedestrian# %d successfully exited crosswalk", pedId);
            return pedId;
        }
    }

    // If the pedestrian was not found, something went wrong
    if (debugLvl > 0)
        printDebug(0, "pedestrian# %d exiting but not found, %d currently still crossing", pedId, pedIdsVector.size());
    return EXITERROR;
}

/**
 * A pedestrian is waiting for vehicles to leave the crosswalk, so assign a waiting ID
 */
int Crosswalk::pedestrianWaiting(const double time)
{
    // Check for crosswalk activity
    checkForReset(time);

    // Assign the pedestrian a unique id
    if (waitIdsVector.empty())
        curWaitId = 0;

    // Increment and assign
    curWaitId++;
    waitIdsVector.push_back(curWaitId);
    if (debugLvl > 0)
        printDebug(1, "pedestrian# %d now waiting, %d vehicles to wait for", curWaitId, vehIdsVector.size());
    return curWaitId;
}
/**
 * Pedestrian with the given ID is done waiting, so remove his ID from the list
 */
int Crosswalk::pedestrianDoneWaiting(const int waitId, const double time)
{
    if (waitId < 0)
        return EXITERROR;

    // Pedestrian done waiting, so update timer
    lastActivity(time);

    // Find the pedestrian's id in the vector, remove it, and return it
    for (std::vector<int>::iterator idIt = waitIdsVector.begin(); idIt != waitIdsVector.end(); idIt++)
    {
        if (waitId == (*idIt))
        {
            waitIdsVector.erase(idIt);
            if (debugLvl > 0)
                printDebug(1, "pedestrian# %d done waiting", waitId);
            return waitId;
        }
    }

    // if the pedestrian was not found, something went wrong
    if (debugLvl > 0)
        printDebug(0, "pedestrian# %d done waiting but not found, %d currently still waiting", waitId, waitIdsVector.size());
    return EXITERROR;
}

/**
 * Attempt to let a new vehicle drive through the crosswalk
 * If no pedestrians are crossing or waiting, assign the vehicle an ID
 * Otherwise, return DONOTENTER
 */
int Crosswalk::enterVehicle(const double time)
{
    // If there are no pedestrians in the crosswalk and no pedestrians waiting, assign this vehicle a unique id
    if (pedIdsVector.empty() && waitIdsVector.empty())
    {
        // Vehicle will be able to enter, so update timer
        lastActivity(time);

        if (vehIdsVector.empty())
            curVehId = 0; // reset ids if this is the only vehicle

        // Increment and assign
        curVehId++;
        vehIdsVector.push_back(curVehId);
        if (debugLvl > 0)
            printDebug(1, "vehicle# %d now driving through crosswalk, %d currently driving through", curVehId, vehIdsVector.size());
        return curVehId;
    }

    // If the crosswalk is currently occupied or being waited for, do not allow the vehicle to enter (must yield to all pedestrians)
    else
    {
        if (debugLvl > 0)
            printDebug(3, "vehicle not allowed to enter, %d pedestrians crossing", pedIdsVector.size());
        return DONOTENTER;
    }
}
/**
 * Vehicle with the given ID has driven through the crosswalk, so remove the ID from the list
 */
int Crosswalk::exitVehicle(const int vehId, const double time)
{
    if (vehId < 0)
        return EXITERROR;

    // Vehicle exiting, so update timer
    lastActivity(time);

    // Find the vehicle's id in the vector, remove it, and return it
    for (std::vector<int>::iterator idIt = vehIdsVector.begin(); idIt != vehIdsVector.end(); idIt++)
    {
        if (vehId == (*idIt))
        {
            vehIdsVector.erase(idIt);
            if (debugLvl > 0)
                printDebug(1, "vehicle# %d successfully exited", vehId);
            return vehId;
        }
    }

    // If the vehicle was not found, something went wrong
    if (debugLvl > 0)
        printDebug(0, "vehicle# %d exiting but not found, %d currently in crosswalk", vehId, vehIdsVector.size());
    return EXITERROR;
}

/**
 * Update the time since the last activity (pedestrian entering/exiting/doneWaiting or vehicle entering/exiting)
 */
void Crosswalk::lastActivity(const double time)
{
    // Print debug msg
    if (debugLvl > 0)
        printDebug(2, "timer updated at %.2f", time);

    // Reset crosswalk if necessary
    lastActivityTime = time;
}

/**
 * Check whether the resetTime has been reached since last activity
 */
void Crosswalk::checkForReset(const double time)
{
    // Ensure that there's been no more than a reasonable time interval since the last action
    double timeDiff = time - lastActivityTime;

    // Print debug msg
    if (debugLvl > 0)
        printDebug(2, "checking for reset, last: %.2f / current: %.2f / diff: %.2f", lastActivityTime, time, timeDiff);

    // Reset crosswalk if necessary
    if (timeDiff >= resetTime)
        resetCrosswalk();
}

/**
 * Reset all pedestrians/vehicles in the crosswalk
 */
void Crosswalk::resetCrosswalk()
{
    // Clear all vehicles and pedestrians in the crosswalk (except pedestrians that are waiting)
    // to ameliorate deadlock situations
    curPedId = 0;
    curVehId = 0;
    pedIdsVector.clear();
    vehIdsVector.clear();

    if (debugLvl > 0)
        printDebug(1, "reset due to inactivity greater than %.2f secs, last activity at %.2f", resetTime, lastActivityTime);
}

/**
 * Print an error/status message
 */
void Crosswalk::printDebug(int lvl, const char *msg, ...)
{
    // If this message is at/below the current debug level, print it
    if (lvl <= debugLvl)
    {
        // Print pedestrian's name
        fprintf(stderr, " Crosswalk %s: ", id.c_str());

        va_list list;
        va_start(list, msg);

        // Print the list
        vfprintf(stderr, msg, list);

        va_end(list);

        // Print a newline
        fprintf(stderr, "\n");
    }
}
