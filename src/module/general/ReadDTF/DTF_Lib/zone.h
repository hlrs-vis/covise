/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/zone.h
 * @brief contains definition of class DTF_Lib::Zone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibZone
 * @brief used to register class DTF_Lib::Zone at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::Zone and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibZone to create new objects of type DTF_Lib::Zone.
 */

/** @class DTF_Lib::Zone
 * @brief contains functions to access informations about zones.
 *
 * \b Description:
 *
 * All zones store zonal data (zd), nodal coordinates, Boundary Condition
 * Records and Volume Condition Records.
 *
 * Groups of cells are associated to Volume Condition Records via the
 * Volume Condition object.
 *
 * Groups of cells are contained in blocks for structured grids and in
 * cell_groups for unstructured grids.
 */

/** @fn DTF_Lib::Zone::Zone();
 * @brief default constructor.
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject for initialization.
 */

/** @fn DTF_Lib::Zone::Zone( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::Zone::~Zone();
 * @brief destructor.
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::Zone::isCartesian ( int simNum,
          int zoneNum,
          bool& result );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param result - result of the query. true if zone is cartesian (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_iscartesian().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Zone::isPoint ( int simNum,
           int zoneNum,
           bool& result );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param result - result of the query. true if zone is point net (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_ispoint().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Zone::isPoly ( int simNum,
          int zoneNum,
          bool& result );
 * @brief check if zone is poly zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param result - result of the query. true if zone is polyzone. (output)
 *
 * @return \c false on error, \c true on success.
*
* \b Description:
 *
 * wraps calls to \c dtf_query_ispoly().
 */

/** @fn bool DTF_Lib::Zone::isStruct ( int simNum,
            int zoneNum,
            bool& result );
 * @brief check if zone is structured zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param result - result of query. true if zone is structured. (output)
 *
 * @return \c false on error, \c true on success.
*
* \b Description:
 *
 * wraps calls to \c dtf_query_isstruct().
 */

/** @fn bool DTF_Lib::Zone::isUnstruct ( int simNum,
         int zoneNum,
         bool& result );
 * @brief check if zone is unstructured
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param result - result of query. true if zone is unstructured. (output)
 *
 * @return \c false on error, \c true on success.
*
* \b Description:
 *
 * wraps calls to \c dtf_query_isunstruct().
 */

/** @fn bool DTF_Lib::Zone::hasBlankingData ( int simNum,
              int zoneNum,
              bool& present );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param present - indicates if blanking data is present. if present, then
value is true. (output)
 *
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_is_blanking_data_present().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Zone::queryCellGroup ( int simNum,
             int zoneNum,
             int groupNum,
             int& key );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param groupNum - cell group number
 *
 * @param key - key (CFD-GEOM entity number) (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_cell_group().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Zone::queryCells ( int simNum,
         int zoneNum,
         vector<int>& cellsOfType,
                        int& numCells);
 * @brief get number of cells for each supported cell type in zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param cellsOfType - vector containing the number of cells for each master
 * cell type (output)
* @param numCells - total number of cells in zone (output)
*
* @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_cells().
 */

/** @fn bool DTF_Lib::Zone::queryCellType ( int simNum,
            int zoneNum,
            int cellNum,
            int& cellType );
 * @brief get type of cell in zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param cellNum - cell number
 *
 * @param cellType - cell type for given cell number (output)
*
* @return \c false on error, \c true on success.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_celltype().
 */

/** @fn bool DTF_Lib::Zone::queryMinMax( int simNum,
         int zoneNum,
         vector<double>& minMax );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param minMax - the grids xyz range of the specified zone; 6 elements (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_minmax_zone_d() and \c dtf_query_minmax_zone_s().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Zone::queryNumCells ( int simNum,
            int zoneNum,
            int& numCells );
 * @brief get number of cells in zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numCells - number of cells in given zone (output)
 *
 * @return \c false on error, \c true on success.
*
* \b Description:
 *
 * wraps calls to \c dtf_query_ncells().
 */

/** @fn bool DTF_Lib::Zone::queryNumCellGroups ( int simNum,
            int zoneNum,
            int& numCellGroups );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numCellGroups - number of cell_groups in this zone (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_ncell_groups().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Zone::queryNumNodes ( int simNum,
            int zoneNum,
            int& numNodes );
 * @brief get number of nodes in zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numNodes - number of nodes in zone (output)
 *
 * @return \c false on error, \c true on success.
*
* \b Description:
 *
 * wraps calls to \c dtf_query_nnodes().
 */

/** @fn bool DTF_Lib::Zone::queryZoneType ( int simNum,
            int zoneNum,
            int& zoneType );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param zoneType - type of the zone (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_zonetype().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Zone::readBlanking ( int simNum,
           int zoneNum,
           int nodeNum,
           vector<int>& blanking );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param nodeNum - node number
 *
 * @param blanking - vector containing the blanking data (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_blanking().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Zone::readCellGroup ( int simNum,
            int zoneNum,
            int groupNum,
            vector<int>& cellNums );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param groupNum - cell group number
 *
 * @param cellNums - vector containing cell numbers (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_cell_group().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Zone::readGrid ( int simNum,
            int zoneNum,
            int nodeNum,
            vector< vector<double> >& coords,
            vector<int>& blanking );
 * @brief get xyz coordinates of grid
 *
 * @param simNum - simulation number
 * @param zoneNum - zoneNumber
 * @param nodeNum - node number ( <1 means all )
 *
* @param coords - vector with X,Y,Z coordinates (output)
* @param blanking - vector with blanking data (output)
*
* @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_grid_d() and \c dtf_read_grid_s().
 */

/** @fn bool DTF_Lib::Zone::readVirtualCellNums ( int simNum,
             int zoneNum,
             vector<int>& cellNums );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zoneNumber
 *
 * @param cellNums - vector with virtual cell nums for zone (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_read_virtual_cellnums().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Zone::readVirtualNodeNums ( int simNum,
             int zoneNum,
             vector<int>& nodeNums );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param nodeNums - vector containing numbers of virtual nodes for the zone
 (output)
 *
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_read_virtual_nodenums().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_ZONE_H_
#define __DTF_LIB_ZONE_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibZone;

class Zone : public LibObject
{
    friend class ClassInfo_DTFLibZone;

    Zone();
    Zone(string className, int objectID);

    static ClassInfo_DTFLibZone classInfo;

public:
    virtual ~Zone();

    virtual bool init();

    Tools::BaseObject *operator()(string className);

    virtual bool setFileHandle(int handle);

    bool isCartesian(int simNum,
                     int zoneNum,
                     bool &result);

    bool isPoint(int simNum,
                 int zoneNum,
                 bool &result);

    bool isPoly(int simNum,
                int zoneNum,
                bool &result);

    bool isStruct(int simNum,
                  int zoneNum,
                  bool &result);

    bool isUnstruct(int simNum,
                    int zoneNum,
                    bool &result);

    bool hasBlankingData(int simNum,
                         int zoneNum,
                         bool &present);

    bool queryCellGroup(int simNum,
                        int zoneNum,
                        int groupNum,
                        int &key);

    bool queryCells(int simNum,
                    int zoneNum,
                    vector<int> &cellsOfType,
                    int &numCells);

    bool queryCellType(int simNum,
                       int zoneNum,
                       int cellNum,
                       int &cellType);

    bool queryMinMax(int simNum,
                     int zoneNum,
                     vector<double> &minMax);

    bool queryNumCells(int simNum,
                       int zoneNum,
                       int &numCells);

    bool queryNumCellGroups(int simNum,
                            int zoneNum,
                            int &numCellGroups);

    bool queryNumNodes(int simNum,
                       int zoneNum,
                       int &numNodes);

    bool queryZoneType(int simNum,
                       int zoneNum,
                       int &zoneType);

    bool readBlanking(int simNum,
                      int zoneNum,
                      int nodeNum,
                      vector<int> &blanking);

    bool readCellGroup(int simNum,
                       int zoneNum,
                       int groupNum,
                       vector<int> &cellNums);

    bool readGrid(int simNum,
                  int zoneNum,
                  int nodeNum,
                  vector<double> &xCoords,
                  vector<double> &yCoords,
                  vector<double> &zCoords,
                  vector<int> &blanking);

    bool readVirtualCellNums(int simNum,
                             int zoneNum,
                             vector<int> &cellNums);

    bool readVirtualNodeNums(int simNum,
                             int zoneNum,
                             vector<int> &nodeNums);
};

CLASSINFO(ClassInfo_DTFLibZone, Zone);
};
#endif

/** EOC */
