/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef RoadSystem_h
#define RoadSystem_h

#include <string>
#include <map>
#include <vector>
#include <ostream>

#include "Element.h"
#include "Road.h"
#include "Junction.h"

class RoadSystem
{
public:
    static RoadSystem *Instance();
    static void Destroy();

    void addRoad(Road *);
    void addJunction(Junction *);

    Road *getRoad(int);
    int getNumRoads();

    Junction *getJunction(int);
    int getNumJunctions();

    void parseOpenDrive(std::string);

protected:
    RoadSystem();

    std::vector<Road *> roadVector;
    std::vector<Junction *> junctionVector;
    std::map<std::string, Road *> roadIdMap;
    std::map<std::string, Junction *> junctionIdMap;

private:
    static RoadSystem *__instance;
};

std::ostream &operator<<(std::ostream &, RoadSystem *);

#endif
