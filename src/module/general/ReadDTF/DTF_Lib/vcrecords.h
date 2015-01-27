/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/vcrecords.h
 * @brief contains definition of class DTF_Lib::VcRecords
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibVcRecords
 * @brief used to register class DTF_Lib::VcRecords at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::VcRecords and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibVcRecords to create new objects of type DTF_Lib::VcRecords.
 */

/** @class DTF_Lib::VcRecords
 * @brief contains access functions for informations about volume condition
 records
 *
 * \b Description:
 *
 * Volume conditions are used to associate groups of cells with arbitrary
 * volumetric data.
 *
 * The volume condition holds
 * - Cell_group number
* - volume condition record number
*
 * The volume condition record stores
 * - category (string)
 * - name (string)
 * - array of \c vcvals (containing name, eval_method, and int, real, or
string data)
*
* Category and name describe the volume condition record. Bcvals store the
* actual data.
*/

/** @fn DTF_Lib::VcRecords::VcRecords();
 * @brief default constructor
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject to do the initialization.
 */

/** @fn DTF_Lib::VcRecords::VcRecords(int* fileHandle);
 * @brief initializes new objects with given file handle
 *
 * @param fileHandle - handle to an open DTF file
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject with given file handle as argument.
 */

/** @fn virtual DTF_Lib::VcRecords::~VcRecords();
 * @brief destructor
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::VcRecords::queryNumRecords( int simNum,
              int zoneNum,
              int& numRecords );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numRecords - number of volume condition records in zone \c zoneNum
 (output)
 *
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_query_nvcrecords().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VcRecords::queryRecord( int simNum,
          int zoneNum,
          int vcNum,
          string& category,
          string& name,
          int& numValues );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param vcNum - volume condition record number
*
* @param category - category of the volume condition record (output)
* @param name - name of the volume condition record (output)
* @param numValues - number of VC values (output)
*
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_vcrecord().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VcRecords::queryEvalData(int simNum,
            int zoneNum,
            int vcNum,
            string name,
            int& numInts,
            int& numReals,
            int& numStrings );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
* @param vcNum - VC record number
* @param name - name of the volume condition record
*
* @param numInts - number of integer values (output)
* @param numReals - number of real values (output)
* @param numStrings - number of string values (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_vcval_eval_data().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VcRecords::queryEvalMethod(int simNum,
              int zoneNum,
              int vcNum,
              string name,
              string& method );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param vcNum - volume condition record number
 * @param name - name of the VC value
*
* @param method - evaluation method of a VC value (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_vcval_eval_method().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VcRecords::queryValName(int simNum,
           int zoneNum,
           int vcNum,
           int valNum,
           string& name );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param vcNum - number of the VC record
 * @param valNum - VC value number
*
* @param name - name of the VC value (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_vcval_name().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VcRecords::readEvalData( int simNum,
           int zoneNum,
           int vcNum,
           string valName,
           EvalData& evalData);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param vcNum - VC record number
 * @param valName - VC value name
*
* @param evalData - object of type EvalData, containing
* - int names and values
* - real names and values
 * - string names and values
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_vcval_eval_data().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VcRecords::readVal(int simNum,
           int zoneNum,
           int vcNum,
           string name,
           string intName,
           int& value );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param vcNum - VC record number
* @param name - VC value name
* @param intName - integer name
*
* @param value - integer value (output)
*
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_vcval_int().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VcRecords::readVal(int simNum,
           int zoneNum,
           int vcNum,
           string name,
           string realName,
           double& value );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param vcNum - VC record number
* @param name - VC value name
* @param realName - real name
*
* @param value - real value (output)
*
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_vcval_real_d() and \c dtf_read_vcval_real_s().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::VcRecords::readVal ( int simNum,
           int zoneNum,
           int vcNum,
           string name,
           string stringName,
           string& value );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param vcNum - VC record number
* @param name - VC value name
* @param stringName - string name
*
* @param value - string value (output)
*
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_vcval_string().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_VCRECORDS_H_
#define __DTF_LIB_VCRECORDS_H_

#include "evaldata.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibVcRecords;

class VcRecords : public LibObject
{
    friend class ClassInfo_DTFLibVcRecords;

    // Associations
    // Attributes
    // Operations
    VcRecords();
    VcRecords(string className, int objectID);

    static ClassInfo_DTFLibVcRecords classInfo;

public:
    virtual ~VcRecords();

    bool queryNumRecords(int simNum,
                         int zoneNum,
                         int &numRecords);

    bool queryRecord(int simNum,
                     int zoneNum,
                     int vcNum,
                     string &category,
                     string &name,
                     int &numValues);

    bool queryEvalData(int simNum,
                       int zoneNum,
                       int vcNum,
                       string name,
                       int &numInts,
                       int &numReals,
                       int &numStrings);

    bool queryEvalMethod(int simNum,
                         int zoneNum,
                         int vcNum,
                         string name,
                         string &method);

    bool queryValName(int simNum,
                      int zoneNum,
                      int vcNum,
                      int valNum,
                      string &name);

    bool readEvalData(int simNum,
                      int zoneNum,
                      int vcNum,
                      string valName,
                      EvalData &evalData);

    bool readVal(int simNum,
                 int zoneNum,
                 int vcNum,
                 string name,
                 string intName,
                 int &value);

    bool readVal(int simNum,
                 int zoneNum,
                 int vcNum,
                 string name,
                 string realName,
                 double &value);

    bool readVal(int simNum,
                 int zoneNum,
                 int vcNum,
                 string name,
                 string stringName,
                 string &value);
};

CLASSINFO(ClassInfo_DTFLibVcRecords, VcRecords);
};
#endif

/** EOC */
