/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/Zone.h
 * @brief contains definition of class DTF::Zone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope.
 */

/** @class DTF::ClassInfo_DTFZone
 * @brief used to register class DTF::Zone at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF::Zone and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFZone to create new objects of type DTF::Zone.
 */

#ifndef __DTF_ZONE_H_
#define __DTF_ZONE_H_

#include "Cell.h"
#include "ZD.h"
#include "../DTF_Lib/structzone.h"
#include "../DTF_Lib/CellTypes.h"
#include "../DTF_Lib/zonedata.h"

namespace DTF
{
class ClassInfo_DTFZone;

class Zone : public Tools::BaseObject
{
    friend class ClassInfo_DTFZone;

private:
    bool structZone;
    bool unstructZone;
    bool polyZone;
    vector<int> dims;

    map<int, Cell *> cells;
    vector<int> numCellsOfType;
    Nodes *nodes;
    ZD *zd;

    Zone();
    Zone(string className, int objectID);

    static ClassInfo_DTFZone classInfo;

    bool readNodes(string fileName,
                   int simNum,
                   int zoneNum,
                   DTF_Lib::Zone *zone);

    bool readZD(int simNum, DTF_Lib::VirtualZone *zone);

public:
    virtual ~Zone();

    virtual bool init();

    bool read(string fileName, int simNum, int zoneNum);

    int getNumCells();
    int getNumNodes();

    Cell *getCell(int cellNum);
    vector<int> getNumCellsOfType();
    vector<int> getCellTypes();

    Nodes *getNodeCoords();
    ZD *getZD();

    bool isStruct();
    bool isUnstruct();
    bool isPoly();
    vector<int> getDims();

    virtual void print();
    virtual void clear();
};

CLASSINFO(ClassInfo_DTFZone, Zone);
};
#endif
