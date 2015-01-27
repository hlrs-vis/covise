/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/bcrecords.h
 * @brief Contains definition of class DTF_Lib::BcRecords
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * completed documentation. Moved static member classInfo to private scope.
 */

/** @class DTF_Lib::ClassInfo_DTFLibBcRecords
 * @brief used to register class DTF_Lib::BcRecords at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::BcRecords and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibBcRecords to create new objects of type DTF_Lib::BcRecords.
 */

/** @class DTF_Lib::BcRecords
 * @brief provides interface to access boundary condition records.
 *
 * Boundary condition records hold data needed to apply boundary conditions
 * to faces. These records are referred to by their number (of type \c int).
 *
 * There are two kinds of faces: boundary faces and interface faces. Each of
 * them are associated to boundary condition records.
 *
 * Associations of faces to records are accomplished by patches for structured
 * grids and by two integer arrays for unstructured grids.
 *
 * \b Patch (structured grid):
 *        holds array of integers that associate each face to a boundary
 *        condition record number.
 *
 * \b integer \b arrays (unstructured grid):
 *        hold global face number and boundary condition record number
 *
 * boundary condition records contain:
 * - type
 * - entity key
 * - name
 * - array of categories
 * - array of bcvals
 */

/** @fn DTF_Lib::BcRecords::BcRecords();
 * @brief default constructor.
 *
 * Calls default constructor of class DTF_Lib::LibObject.
 */

/** @fn DTF_Lib::BcRecords::BcRecords( string className, int objectID );
 * @brief constructor called with argument for initialization
 *
 * @param className - name of the class
 * @param objectID - file handle to open DTF file
 *
 * \b Description:
 *
 * The objectID is used by the class manager when creating new objects and
 * deleting them.
 *
 * The className is needed by the statistic manager to manage statistics about
 * objects of this class.
 */

/** @fn virtual DTF_Lib::BcRecords::~BcRecords();
 * @brief default destructor
 *
 * Called when objects of this class are destroyed.
 */

/** @fn bool DTF_Lib::BcRecords::queryCategory(int simNum,
int zoneNum,
int bcNum,
int catNum,
string& name,
string& value);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param bcNum - boundary condition record number
* @param catNum - category number
* @param name - category name (output)
* @param value - category value (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_bc_category().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcRecords::queryCategoryVal( int simNum,
               int zoneNum,
               int bcNum,
               string name,
               string& value );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param bcNum - record number
 * @param name - category name (output)
* @param value - category value (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_bc_category_value().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 *
 */

/** @fn bool DTF_Lib::BcRecords::queryEvalData ( int simNum,
            int zoneNum,
            int bcNum,
            string valueName,
            int& numInts,
            int& numReals,
            int& numStrings );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
* @param bcNum - record number
* @param valueName - BC value name
*
* @param numInts - number of integers (output)
* @param numReals - number reals (output)
* @param numStrings - number of strings (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_bcval_eval_data().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcRecords::queryEvalMethod ( int simNum,
              int zoneNum,
              int bcNum,
              string name,
              string& evalMethod );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param bcNum - record number
 * @param name - BC value name
*
* @param evalMethod - evaluation method.
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_bcval_eval_method().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcRecords::queryNumRecords ( int simNum,
              int zoneNum,
              int& numRecords );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param numRecords - number of boundary condition records in given
 zone/simulation
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_query_nbcrecords().
*
* @attention You'll have to implement this function if you intend to use it. A
* warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcRecords::queryRecord ( int simNum,
          int zoneNum,
          int bcNum,
          int& key,
          string& type,
          string& name,
          int& numCat,
          int& numVals );
 * @brief not implemented.
 *
 * @param simNum - simulation number
* @param zoneNum - zone number
* @param bcNum - boundary condition record number
*
* @param key - key (output)
* @param type - type of boundary condition record (output)
* @param name - name of boundary condition record (output)
* @param numCat - number of BC categories (output)
 * @param numVals - number of BC values (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_bcrecord().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcRecords::queryValName ( int simNum,
           int zoneNum,
           int bcNum,
           int valNum,
           string& name );
 * @brief not implemented.
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param bcNum - boundary condition record number
 * @param valNum - value number
*
* @param name - name of the BC value (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_bcval_name().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcRecords::readEvalData ( int simNum,
           int zoneNum,
           int bcNum,
           string name,
           EvalData& evalData);
 * @brief not implemented.
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param bcNum - BC record number
 * @param name - BC value name
*
* @param evalData - encapsulated evaluation data (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_bcval_eval_data().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcRecords::readVal ( int simNum,
           int zoneNum,
           int bcNum,
           string name,
           string intName,
           int& value );
 * @brief not implemented.
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param bcNum - BC record number
* @param name - BC value name
* @param intName - name of the integer value
*
* @param value - integer value (output)
*
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_bcval_int().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcRecords::readVal ( int simNum,
           int zoneNum,
           int bcNum,
           string name,
           string realName,
           double& value );
 * @brief not implemented.
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param bcNum - BC record number
* @param name - BC value name
* @param realName - name of the double value
*
* @param value - double value (output)
*
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_bcval_real_d() and \c dtf_read_bcval_real_s().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcRecords::readVal ( int simNum,
           int zoneNum,
           int bcNum,
           string name,
           string stringName,
           string& value );
 * @brief not implemented.
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param bcNum - BC record number
* @param name - BC value name
* @param stringName - name of the string
*
* @param value - string value (output)
*
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_string().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @var static ClassInfo_DTFLibBcRecords DTF_Lib::BcRecords::classInfo;
 * @brief static object used to register class at class manager
 *
 * \b Description:
 *
 * Calls constructor of generated class ClassInfo_DTFLibBcRecords to register
 * class DTF_Lib::BcRecords at class manager.
 */

/** EOD */

/** BOC (begin of code) */
#ifndef __DTF_LIB_BCRECORDS_H_
#define __DTF_LIB_BCRECORDS_H_

#include "evaldata.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibBcRecords;

class BcRecords : public DTF_Lib::LibObject
{
    /** only class allowed to create new objects */
    friend class ClassInfo_DTFLibBcRecords;

private:
    BcRecords();
    BcRecords(string className, int objectID);

    static ClassInfo_DTFLibBcRecords classInfo;

public:
    virtual ~BcRecords();

    bool queryCategory(int simNum,
                       int zoneNum,
                       int bcNum,
                       int catNum,
                       string &name,
                       string &value);

    bool queryCategoryVal(int simNum,
                          int zoneNum,
                          int bcNum,
                          string name,
                          string &value);

    bool queryEvalData(int simNum,
                       int zoneNum,
                       int bcNum,
                       string valueName,
                       int &numInts,
                       int &numReals,
                       int &numStrings);

    bool queryEvalMethod(int simNum,
                         int zoneNum,
                         int bcNum,
                         string name,
                         string &evalMethod);

    bool queryNumRecords(int simNum,
                         int zoneNum,
                         int &numRecords);

    bool queryRecord(int simNum,
                     int zoneNum,
                     int bcNum,
                     int &key,
                     string &type,
                     string &name,
                     int &numCat,
                     int &numVals);

    bool queryValName(int simNum,
                      int zoneNum,
                      int bcNum,
                      int valNum,
                      string &name);

    bool readEvalData(int simNum,
                      int zoneNum,
                      int bcNum,
                      string name,
                      EvalData &evalData);

    bool readVal(int simNum,
                 int zoneNum,
                 int bcNum,
                 string name,
                 string intName,
                 int &value);

    bool readVal(int simNum,
                 int zoneNum,
                 int bcNum,
                 string name,
                 string realName,
                 double &value);

    bool readVal(int simNum,
                 int zoneNum,
                 int bcNum,
                 string name,
                 string stringName,
                 string &value);
};

CLASSINFO(ClassInfo_DTFLibBcRecords, BcRecords);
};
#endif

/** EOC */
