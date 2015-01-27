/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/Sim.h
 * @brief contains definition of class DTF::Sim
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF::ClassInfo_DTFSim
 * @brief used to register class DTF::Sim at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF::Sim and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFSim to create new objects of type DTF::Sim.
 */

#ifndef __DTF_SIM_H_
#define __DTF_SIM_H_

#include "Zone.h"
#include "../DTF_Lib/sim.h"

namespace DTF
{
class ClassInfo_DTFSim;

class Sim : public Tools::BaseObject
{
    friend class ClassInfo_DTFSim;

private:
    string simDescr;
    int numSDs;

    map<int, Zone *> zones;

    Sim();
    Sim(string className, int objectID);

    static ClassInfo_DTFSim classInfo;

public:
    virtual ~Sim();

    virtual bool init();
    bool read(string fileName, int simNum);

    string getSimDescr();
    int getNumSDs();

    Zone *getZone(int zoneNum);
    int getNumZones();
    void print();
    void clear();
};

CLASSINFO(ClassInfo_DTFSim, Sim);
};
#endif
