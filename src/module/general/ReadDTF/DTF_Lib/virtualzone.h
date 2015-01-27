/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/virtualzone.h
 * @brief contains definition of class DTF_Lib::VirtualZone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibVirtualZone
 * @brief used to register class DTF_Lib::VirtualZone at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::VirtualZone and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibVirtualZone to create new objects of type DTF_Lib::VirtualZone.
 */

/** @class DTF_Lib::VirtualZone
 * @brief contains access functions for informations related to virtual zones
 *
 * \b Description:
 *
 * The virtual zone is created by invoking the connectivity API of CFD-DTF
 * with a zone number of zero. This mode of operation concatenates structured
 * and/or unstructured grid zones into a single, virtual unstructured zone.
 */

/** @fn DTF_Lib::VirtualZone::VirtualZone();
 * @brief default constructor
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject for initialization.
 */

/** @fn DTF_Lib::VirtualZone::VirtualZone( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn bool DTF_Lib::VirtualZone::~VirtualZone();
 * @brief destructor
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::VirtualZone::queryBCrecNum(int simNum,
            int zoneNum,
            int bcRecNum,
                           int& vzBcRecNum);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param bcRecNum - BC record number
 *
 * @param vzBcRecNum - BC record number in virtual zone
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_vz_bcrec_num().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VirtualZone::queryNumVZDs(int simNum,
           int& numVZDs );
 * @brief get number of data arrays attached to VZ
 *
 * @param simNum - simulation number
 *
 * @param numVZDs - number of virtual zones in simulation (output)
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
*
 * wraps calls to \c dtf_query_nvzds().
 */

/** @fn bool DTF_Lib::VirtualZone::queryNumVZDsOfTopotype ( int simNum,
                int topotype,
                int& numVZDs );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param topotype - topological type
 *
 * @param numVZDs - number of virtual zone data arrays of given topotype (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_nvzds_of_topotype().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VirtualZone::queryVZDbyName ( int simNum,
             string name,
             DataElement& vzdInfo);
 * @brief return info about vzd data array with given name
 *
 * @param simNum - simulation number
 * @param name - name of the data array
 *
 * @param vzdInfo - information about the virtual zones data array, encapsulated
 * in object of type DTF_Lib::DataElement. (output)
 *
* @return \c false on error, \c true on success.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_vzd_by_name().
 */

/** @fn bool DTF_Lib::VirtualZone::queryVZDbyNum ( int simNum,
            int vzdNum,
            DataElement& vzdInfo);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param vzdNum - number of the virtual zone data array
 *
 * @param vzdInfo - information about the virtual zones data array, encapsulated
 * in object of type DTF_Lib::DataElement. (output)
 *
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_vzd_by_num().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VirtualZone::readVZDbyName ( int simNum,
            string name,
            vector<void*>& data,
            int& datatype );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param name - name of the virtual zone data array
 *
 * @param data - vector with virtual zone data (output)
 * @param datatype - datatype of virtual zone data (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_vzd_by_name().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VirtualZone::readVZDbyNum ( int simNum,
           int vzdNum,
           vector<void*>& data,
           int& datatype);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param vzdNum - instance number of data array in virtual zone
 *
 * @param data - vector with data from virtual zone data array (output)
 * @param datatype - datatype of virtual zone data array (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_vzd_by_num().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VirtualZone::readVZDNumsOfTopotype ( int simNum,
               int topotype,
               vector<int>& vzdNums );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param topotype - topological type of virtual zone data
 *
 * @param vzdNums - vector containing vzd numbers (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_read_vzdnums_of_topotype().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_VIRTUALZONE_H_
#define __DTF_LIB_VIRTUALZONE_H_

#include "dataelement.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibVirtualZone;

class VirtualZone : public LibObject
{
    friend class ClassInfo_DTFLibVirtualZone;

    // Associations
    // Attributes
    // Operations
    VirtualZone();
    VirtualZone(string className, int objectID);

    static ClassInfo_DTFLibVirtualZone classInfo;

public:
    virtual ~VirtualZone();

    bool queryBCrecNum(int simNum,
                       int zoneNum,
                       int bcRecNum,
                       int &vzBcRecNum);

    bool queryNumVZDs(int simNum,
                      int &numVZDs);

    bool queryNumVZDsOfTopotype(int simNum,
                                int topotype,
                                int &numVZDs);

    bool queryVZDbyName(int simNum,
                        string name,
                        DataElement &vzdInfo);

    bool queryVZDbyNum(int simNum,
                       int vzdNum,
                       DataElement &vzdInfo);

    bool queryVZDNames(int simNum,
                       vector<string> &names);

    bool readVZDbyName(int simNum,
                       string name,
                       vector<int> &data);

    bool readVZDbyName(int simNum,
                       string name,
                       vector<double> &data);

    bool readVZDbyName(int simNum,
                       string name,
                       vector<string> &data);

    bool readVZDbyNum(int simNum,
                      int vzdNum,
                      vector<void *> &data,
                      int &datatype);

    bool readVZDNumsOfTopotype(int simNum,
                               int topotype,
                               vector<int> &vzdNums);
};

CLASSINFO(ClassInfo_DTFLibVirtualZone, VirtualZone);
};
#endif

/** EOC */
