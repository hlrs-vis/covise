/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/zonedata.h
 * @brief contains definition of class DTF_Lib::ZoneData
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibZoneData
 * @brief used to register class DTF_Lib::ZoneData at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::ZoneData and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibZoneData to create new objects of type DTF_Lib::ZoneData.
 */

/** @class DTF_Lib::ZoneData
 * @brief contains functions to access zone data.
 */

/** @fn DTF_Lib::ZoneData::ZoneData();
 * @brief default constructor
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject for initialization.
 */

/** @fn DTF_Lib::ZoneData::ZoneData( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::ZoneData::~ZoneData();
 * @brief destructor
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::ZoneData::queryMinMax ( int simNum,
          int zoneNum,
          vector<double>& minMax );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param minMax - vector with grids xyz range for zonal data (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_zd_minmax_by_name().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneData::queryNumZDs ( int simNum,
          int zoneNum,
          int& numZDs );
 * @brief get number of zone data arrays in zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numZDs - number of data arrays attached to zone (output)
 *
 * @return \c false on error, \c true on success.
*
* \b Description:
 *
 * wraps calls to \c dtf_query_nzds().
 */

/** @fn bool DTF_Lib::ZoneData::queryNumZDsOfTopotype ( int simNum,
               int zoneNum,
               int topotype,
               int& numZDs );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param topotype - topological type
 *
 * @param numZDs - number of zone data arrays for given topotype (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_nzds_of_topotype().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneData::queryZDbyName ( int simNum,
            int zoneNum,
            string name,
            DataElement& zdInfo);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param name - name of the zone data array
 *
 * @param zdInfo - information about the zone data array, encapsulated into
* object of type DTF_Lib::DataElement. (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_zd_by_name().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneData::queryZDbyNum ( int simNum,
           int zoneNum,
           int dataNum,
           DataElement& zdInfo);
 * @brief query zone data for given zd number
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param dataNum - number of data array
 *
 * @param zdInfo - information about the zone data array, encapsulated into
* object of type DTF_Lib::DataElement. (output)
*
* @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_zd_by_num().
 */

/** @fn bool DTF_Lib::ZoneData::queryZDNames ( int simNum,
           int zoneNum,
                          vector<string&> names);
 * @brief query zone data for given zd number
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param names - vector of string containing the data names for this zone
 *
 * @return \c false on error, \c true on success.
*
* \b Description:
 *
 * makes subsequent calls to queryNumZDs() and queryZDbyNum() to acquire the
 * data list.
 */

/** @fn bool DTF_Lib::ZoneData::readNumZDsOfTopotype ( int simNum,
              int zoneNum,
              int topotype,
              vector<int>& nums );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param topotype - topological type of zone data array
 *
 * @param nums - vector of data array numbers (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_zdnums_of_topotype().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneData::readZDbyName ( int simNum,
           int zoneNum,
           string name,
           int elementNum,
           vector<int>& data );
 * @brief read int data array attached to a zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param name - name of the zone data array
 * @param elementNum - element to read ( <0 means all )
*
* @param data - vector with elements of the zone data array (output)
*
* @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_zd_by_name().
 */

/** @fn bool DTF_Lib::ZoneData::readZDbyName ( int simNum,
           int zoneNum,
           string name,
           int elementNum,
           vector<double>& data );
 * @brief read double data array attached to a zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param name - name of the zone data array
 * @param elementNum - element to read ( <0 means all )
*
* @param data - vector with elements of the zone data array (output)
*
* @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_zd_by_name().
 */

/** @fn bool DTF_Lib::ZoneData::readZDbyName ( int simNum,
           int zoneNum,
           string name,
           int elementNum,
           vector<string>& data );
 * @brief read string data array attached to a zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param name - name of the zone data array
 * @param elementNum - element to read ( <0 means all )
*
* @param data - vector with elements of the zone data array (output)
*
* @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_zd_by_name().
 */

/** @fn bool DTF_Lib::ZoneData::readZDbyNum ( int simNum,
          int zoneNum,
          int dataNum,
          int elementNum,
          vector<int>& data );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param dataNum - number of zone data array
 * @param elementNum - element to read ( < 0 means all )

* @param data - vector with elements of the data array (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_zd_by_num().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneData::readZDbyNum ( int simNum,
          int zoneNum,
          int dataNum,
          int elementNum,
          vector<double>& data );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param dataNum - number of zone data array
 * @param elementNum - element to read ( < 0 means all )

* @param data - vector with elements of the data array (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_zd_by_num().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneData::readZDbyNum ( int simNum,
          int zoneNum,
          int dataNum,
          int elementNum,
          vector<string>& data );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param dataNum - number of zone data array
 * @param elementNum - element to read ( < 0 means all )

* @param data - vector with elements of the data array (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_zd_by_num().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_ZONEDATA_H_
#define __DTF_LIB_ZONEDATA_H_

#include "dataelement.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibZoneData;

class ZoneData : public LibObject
{
    friend class ClassInfo_DTFLibZoneData;

    ZoneData();
    ZoneData(string className, int objectID);

    static ClassInfo_DTFLibZoneData classInfo;

public:
    virtual ~ZoneData();

    bool queryMinMax(int simNum,
                     int zoneNum,
                     vector<double> &minMax);

    bool queryNumZDs(int simNum,
                     int zoneNum,
                     int &numZDs);

    bool queryNumZDsOfTopotype(int simNum,
                               int zoneNum,
                               int topotype,
                               int &numZDs);

    bool queryZDbyName(int simNum,
                       int zoneNum,
                       string name,
                       DataElement &zdInfo);

    bool queryZDbyNum(int simNum,
                      int zoneNum,
                      int dataNum,
                      DataElement &zdInfo);

    bool queryZDNames(int simNum,
                      int zoneNum,
                      vector<string> &names);

    bool readNumZDsOfTopotype(int simNum,
                              int zoneNum,
                              int topotype,
                              vector<int> &nums);

    bool readZDbyName(int simNum,
                      int zoneNum,
                      string name,
                      int elementNum,
                      vector<int> &data);

    bool readZDbyName(int simNum,
                      int zoneNum,
                      string name,
                      int elementNum,
                      vector<double> &data);

    bool readZDbyName(int simNum,
                      int zoneNum,
                      string name,
                      int elementNum,
                      vector<string> &data);

    bool readZDbyNum(int simNum,
                     int zoneNum,
                     int dataNum,
                     int elementNum,
                     vector<int> &data);

    bool readZDbyNum(int simNum,
                     int zoneNum,
                     int dataNum,
                     int elementNum,
                     vector<double> &data);

    bool readZDbyNum(int simNum,
                     int zoneNum,
                     int dataNum,
                     int elementNum,
                     vector<string> &data);
};

CLASSINFO(ClassInfo_DTFLibZoneData, ZoneData);
};
#endif

/** EOC */
