/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef Crosswalk_h
#define Crosswalk_h

#include "Element.h"

#include <osg/Node>
#include <osg/Group>

#include <string>
#include <limits.h>
#include <stdarg.h> // for printDebug()'s variable argument list

class VEHICLEUTILEXPORT Crosswalk : public Element
{
public:
    static const int DONOTENTER;
    static const int EXITERROR;

    Crosswalk(const std::string &id_, const std::string &name_ = "unnamed-crosswalk", const double s_ = 0.0, const double length_ = 0.0, const double crossProb_ = 0.5, const double resetTime_ = 20.0, const std::string &type_ = "crosswalk", int dbg_ = 0);

    double getS() const
    {
        return s;
    }
    double getLength() const
    {
        return length;
    }
    double getCrossProb() const
    {
        return crossProb;
    }

    void setValidity(const int from, const int to);
    std::pair<int, int> getValidity() const;

    int enterPedestrian(const double time);
    int exitPedestrian(const int id, const double time);

    int pedestrianWaiting(const double time);
    int pedestrianDoneWaiting(const int id, const double time);

    int enterVehicle(const double time);
    int exitVehicle(const int id, const double time);

    void resetCrosswalk();

protected:
    double lastActivityTime;

    std::string id;
    std::string name;

    double s;
    double length;
    double crossProb;
    double resetTime;

    std::string type;

    int validFromLane;
    int validToLane;

    int curPedId;
    std::vector<int> pedIdsVector;

    int curWaitId;
    std::vector<int> waitIdsVector;

    int curVehId;
    std::vector<int> vehIdsVector;

    int debugLvl;

private:
    void lastActivity(const double time);
    void checkForReset(const double time);

    void printDebug(int lvl, const char *msg, ...);
};

#endif
