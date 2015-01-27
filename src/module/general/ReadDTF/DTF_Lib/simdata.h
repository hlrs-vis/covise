/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/simdata.h
 * @brief contains definition of class DTF_Lib::SimData
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibSimData
 * @brief used to register class DTF_Lib::SimData at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::SimData and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibSimData to create new objects of type DTF_Lib::SimData.
 */

/** @class DTF_Lib::SimData
 * @brief contains functions to access simulation data.
 */

/** @fn DTF_Lib::SimData::SimData();
 * @brief default constructor
 *
 * \b Description:
 *
 * Calls default constructor of DTF_Lib::SimData.
 */

/** @fn DTF_Lib::SimData::SimData( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::SimData::~SimData();
 * @brief default destructor
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::SimData::queryNumSDsOfTopotype ( int simNum,
               int topotype,
               int& numSDs );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param topotype - topological type.
 * - DTF_GNRL_TOPO 0 (data not associated to geometric entity)
 * - DTF_NODE_TOPO 1 (data is associated with mesh nodes)
 * - DTF_EDGE_TOPO 2 (data is associated with mesh edges)
 * - DTF_FACE_TOPO 3 (data is associated with mesh faces)
* - DTF_CELL_TOPO 4 (data is associated with mesh cells)
*
 * @param numSDs - number of data arrays for given topotype (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_nsds_of_topotype().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::SimData::querySDbyName(int simNum,
            string name,
            DataElement& sdInfo);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param name - name of simulation data
 *
 * @param sdInfo - informations about the simulation data array, encapsulated
 * in object of type DTF_Lib::DataElement. (output)
 *
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_sd_by_name().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::SimData::querySDbyNum(int simNum,
           int dataNum,
           DataElement& sdInfo);
 * @brief get info about a data array attached to a simulation
 *
 * @param simNum - simulation number
 * @param dataNum - number of the data array
 *
 * @param sdInfo - informations about the simulation data array, encapsulated
 * in object of type DTF_Lib::DataElement. (output)
 *
* @return \c false on error, \c true on success.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_sd_by_num().
 */

/** @fn bool DTF_Lib::SimData::readNumSDsOfTopotype(int simNum,
              int topotype,
              vector<int>& nums );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param topotype - topological type of the simulation data arrays
 *
 * @param nums - vector containing the numbers of data arrays for given
 topotype (output)
 *
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_query_sdnums_by_topotype().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::SimData::readSDbyName ( int simNum,
           string name,
           int elementNum,
           int& datatype,
           vector<void*>& data );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param name - name of the simulation data array to read
 * @param elementNum - element number to read from array ( <=0 for all )
 * @param datatype - datatype of the elements
*
* @param data - vector containing the simulation data
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_sd_by_name().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::SimData::readSDbyNum ( int simNum,
int dataNum,
int elementNum,
int& datatype,
vector<void*>& data );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param dataNum - number of the simulation data array to read
 * @param elementNum - element number to read from array ( <=0 for all )
 * @param datatype - datatype of the elements
*
* @param data - vector containing the simulation data
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_sd_by_num().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_SIMDATA_H
#define __DTF_LIB_SIMDATA_H

#include "dataelement.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibSimData;

class SimData : public LibObject
{
    friend class ClassInfo_DTFLibSimData;

private:
    SimData();
    SimData(string className, int objectID);

    static ClassInfo_DTFLibSimData classInfo;

public:
    virtual ~SimData();

    bool queryNumSDsOfTopotype(int simNum,
                               int topotype,
                               int &numSDs);

    bool querySDbyName(int simNum,
                       string name,
                       DataElement &sdInfo);

    bool querySDbyNum(int simNum,
                      int dataNum,
                      DataElement &sdInfo);

    bool readNumSDsOfTopotype(int simNum,
                              int topotype,
                              vector<int> &nums);

    bool readSDbyName(int simNum,
                      string name,
                      int elementNum,
                      int &datatype,
                      vector<void *> &data);

    bool readSDbyNum(int simNum,
                     int dataNum,
                     int elementNum,
                     int &datatype,
                     vector<void *> &data);
};

CLASSINFO(ClassInfo_DTFLibSimData, SimData);
};
#endif

/** EOC */
