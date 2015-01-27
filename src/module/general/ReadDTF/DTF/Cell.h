/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF/Cell.h
 * @brief contains definition of class DTF::Cell
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 28.10.2003
 * created
 * @date 5.11.2003
 * move static member classInfo to private scope.
 */

/** @class DTF::ClassInfo_DTFCell
 * @brief used to register class DTF::Cell at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF::Cell and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFCell to create new objects of type DTF::Cell.
 */

/** @class DTF::Cell;
 * @brief represents data associated with cell in structured grid zone
 *
 * This class provides access to data associated with cells in structured grid
 * zones.
 *
 * Each cell contains a specific number of nodes. In order to reduce memory
 * consumption objects of class DTF::Cell contain only a list of node numbers
 * associated with the cell.
 */

/** @fn DTF::Cell::Cell();
 * @brief default constructor
 *
 * Adds object statistic to statistic manager.
 */

/** @fn DTF::Cell::Cell(string className, int objectID);
 * @brief creates new object with given class name and object id
 *
 * @param className - name of the class
 * @param objectID - unique identifier for the created object
 *
 * New objects are created by the class manager through calls like this:
 * @code
 * DTF::Cell* cell = Tools::ClassManager::getInstance()->getObject("DTF::Cell");
 * @endcode
 *
 * The objectID is assigned and used by the class manager to identify the
 * object.
 */

/** @fn DTF::Cell::~Cell();
 * @brief clears memory when object is destroyed
 *
 * Clears node list and tells statistic manager that object has been deleted.
 */

/** @fn bool DTF::Cell::read(string fileName,
int simNum,
int zoneNum,
int cellNum);
 * @brief reads data associated with cell in structured grid zone.
 *
 * @param fileName - path to DTF file
 * @param simNum - simulation number which contains the structured grid
 * @param zoneNum - structured grid zone number
 * @param cellNum - number of cell in structured grid zone
 *
* @return \c true if data could be read. \c false on error.
*
* This function reads the cell->node data array associated with the given cell
 * number in the structured grid zone. The cell->node array contains the node
 * numbers associated with the cell and is stored as node list in DTF::Cell::nodes.
 */

/** @fn vector<int> DTF::Cell::getNodeNumbers();
 * @brief get list of node numbers
 *
 * @return vector containing a list of node numbers associated with the cell.
 *
 * Returns a list of node numbers associated with the cell. Coordinates of that
 * nodes are stored in the corresponding DTF::Zone.
 */

/** @fn int DTF::Cell::getNumNodes();
 * @brief get number of nodes in cell
 *
 * @return number of nodes in cell. 0 if there are none.
 */

/** @fn virtual void DTF::Cell::clear();
 * @brief clears memory occupied by cell.
 *
 * Clears node list and sets cellType to \c 0.
 */

/** @fn virtual bool DTF::Cell::init();
 * @brief initialization of object
 *
 * Called by class manager to initialize new objects of type DTF::Cell.
 *
 * Clears node list ( just to be sure ;) )
 */

/** @fn virtual void DTF::Cell::print();
 * @brief print contents of cell.
 *
 * Prints number of nodes and node numbers associated with the cell to stdout.
 */

/**@var vector<int> DTF::Cell::nodes;
 * @brief list of node numbers contained in cell.
 */

/**@var int DTF::Cell::cellType;
 * @brief type of DTF cell.
 *
 * Available types are:
 * - 1: Triangle
 * - 2: Quadrilateral
 * - 3: Tetrahedron
 * - 4: Pyramid
 * - 5: Prism
 * - 6: Hexahedron
 *
 * @note The cell type must be incremented by 1 before use by Covise modules
 * since Covise knows an additional cell type (point). Therefore the first
 * cell type recognized by Covise is Point (1) followed by Triangle (2) and
 * so on.
 */

/**@var static ClassInfo_DTFCell DTF::Cell::classInfo;
 * @brief used to register class DTF::Cell.
 *
 * This static object is used to register class DTF::Cell at the class manager.
 * The class manager is responsible for creation and deletion of new objects
 * of class DTF::Cell.
 */

#ifndef __DTF_CELL_H_
#define __DTF_CELL_H_

#include "Node.h"
#include "../DTF_Lib/mesh.h"

namespace DTF
{
class ClassInfo_DTFCell;

class Cell : public Tools::BaseObject
{
    /** This class is the only class allowed to create new objects of DTF::Cell
       */
    friend class ClassInfo_DTFCell;

private:
    vector<int> nodes;
    int cellType;

    Cell();
    Cell(string className,
         int objectID);

    static ClassInfo_DTFCell classInfo;

public:
    virtual ~Cell();

    bool read(string fileName,
              int simNum,
              int zoneNum,
              int cellNum);

    vector<int> getNodeNumbers();
    int getNumNodes();

    virtual void clear();
    virtual bool init();
    virtual void print();
};

CLASSINFO(ClassInfo_DTFCell, Cell);
};
#endif
