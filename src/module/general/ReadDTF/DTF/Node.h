/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/Node.h
 * @brief contains definition of class DTF::Node and struct DTF::Coords
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope.
 */

/** @struct DTF::Coords
 * @brief struct containing 3D coordinates
 *
 * Simplifies setting and getting of 3D coordinates of points.
 */

/** @fn DTF::Coords::Coords()
 * @brief sets default coordinates
 *
 * All coordinates are initialized to \c 0.
 */

/** @fn DTF::Coords::~Coords()
 * @brief default destructor
 *
 * Does nothing.. Only provided for completeness.
 */

/** @var DTF::Coords::x
 * @brief x coordinate of a point
 */

/** @var DTF::Coords::y
 * @brief y coordinate of a point
 */

/** @var DTF::Coords::z
 * @brief z coordinate of a point
 */

/** @class DTF::ClassInfo_DTFNode
 * @brief used to register class DTF::Node at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF::Node and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFNode to create new objects of type DTF::Node.
 */

#ifndef __DTF_NODE_H_
#define __DTF_NODE_H_

#include "../DTF_Lib/libif.h"
#include "../DTF_Lib/zone.h"

namespace DTF
{
class ClassInfo_DTFNodes;

struct Coords
{
    double x;
    double y;
    double z;
    Coords()
    {
        x = y = z = 0.0;
    };
    ~Coords(){};
};

class Nodes : public Tools::BaseObject
{
    friend class ClassInfo_DTFNodes;

private:
    map<int, Coords *> coords;

    Nodes();
    Nodes(string className, int objectID);
    static ClassInfo_DTFNodes classInfo;

public:
    virtual ~Nodes();

    virtual bool init();
    bool read(string fileName, int simNum, int zoneNum);

    bool getCoords(int nodeNum, Coords &value);
    int getNumCoords();
    void print();
    virtual void clear();
};

CLASSINFO(ClassInfo_DTFNodes, Nodes);
};
#endif
