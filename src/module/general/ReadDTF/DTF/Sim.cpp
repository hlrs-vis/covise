/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/Sim.h
 * @brief contains implementation of class DTF::Sim
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 */
#include "Sim.h"

using namespace DTF;

CLASSINFO_OBJ(ClassInfo_DTFSim, Sim, "DTF::Sim", INT_MAX);

Sim::Sim()
{
    INC_OBJ_COUNT(getClassName());
}

Sim::Sim(string className, int objectID)
    : Tools::BaseObject(className, objectID)
{
    simDescr = "";
    numSDs = 0;
    zones.clear();

    INC_OBJ_COUNT(getClassName());
}

Sim::~Sim()
{
    zones.clear();

    DEC_OBJ_COUNT(getClassName());
}

bool Sim::init()
{
    zones.clear();

    return true;
}

bool Sim::read(string fileName, int simNum)
{
    clear();
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    DTF_Lib::LibIF *libif = (DTF_Lib::LibIF *)cm->getObject("DTF_Lib::LibIF");
    DTF_Lib::Sim *sim = (DTF_Lib::Sim *)(*libif)("Sim");
    string store;
    int sdNumData = 0;
    int numZones = 0;

    if (sim->querySimDescr(simNum, store))
        this->simDescr = store;
    if (sim->queryNumSDs(simNum, sdNumData))
        this->numSDs = sdNumData;

    if (sim->queryNumZones(simNum, numZones))
    {
        Zone *zone = (DTF::Zone *)cm->getObject("DTF::Zone");

        if (!zone->read(fileName, simNum, 0))
        {
            cm->deleteObject(zone->getID());
            return false;
        }

#ifdef DEBUG_MODE
        cout << "sim #" << simNum << ": adding zone #" << 0 << endl;
#endif

        if (zones.find(0) == zones.end())
            zones.insert(pair<int, Zone *>(0, zone));
        else
            cm->deleteObject(zone->getID());

        if (zone != NULL)
            zone = NULL;
    }

    return true;
}

Zone *Sim::getZone(int zoneNum)
{
    Zone *zone = NULL;
    map<int, Zone *>::iterator zoneIterator = zones.find(zoneNum);

    if (zoneIterator != zones.end())
        zone = zoneIterator->second;

    return zone;
}

int Sim::getNumZones()
{
    return this->zones.size();
}

string Sim::getSimDescr()
{
    return this->simDescr;
}

int Sim::getNumSDs()
{
    return this->numSDs;
}

void Sim::print()
{
    map<int, Zone *>::iterator zoneIterator;

    cout << this->getSimDescr() << endl;
    cout << "\t" << this->getNumSDs() << " data arrays " << endl;

    if (this->getNumZones())
    {
        cout << "\t" << this->getNumZones() << " zone(s):" << endl;

        zoneIterator = this->zones.begin();

        while (zoneIterator != this->zones.end())
        {
            cout << "\t - zone " << zoneIterator->first << ": ";

            Zone *zone = zoneIterator->second;

            if (zone != NULL)
                zone->print();
            else
            {
                cout << endl;
            }

            ++zoneIterator;
        }
    }
}

void Sim::clear()
{
    Tools::ClassManager *cm = Tools::ClassManager::getInstance();
    map<int, Zone *>::iterator zoneIterator = zones.begin();

    while (zoneIterator != zones.end())
    {
        Zone *zone = zoneIterator->second;

        if (zone != NULL)
        {
            zone->clear();
            cm->deleteObject(zone->getID());

            zone = NULL;
        }

        ++zoneIterator;
    }
}
