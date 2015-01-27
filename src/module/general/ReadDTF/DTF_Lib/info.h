/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/info.h
 * @brief contains definition of class DTF_Lib::Info
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibInfo
 * @brief used to register class DTF_Lib::Info at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::Info and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibInfo to create new objects of type DTF_Lib::Info.
 */

/** @class DTF_Lib::Info
 * @brief contains access functions for some general DTF file info.
 */

/** @fn DTF_Lib::Info::Info();
 * @brief default constructor
 *
 * \b Description:
 *
 * calls default constructor of class DTF_Lib::LibObject for initialization.
 */

/** @fn DTF_Lib::Info::Info( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::Info::~Info();
 * @brief default destructor
 *
 * \b Description:
 *
 * Called when objects of class DTF_Lib::Info are destroyed.
 */

/** @fn bool DTF_Lib::Info::queryApplication(string& application);
 * @brief not implemented
 *
 * @param application - reference to a string holding the name of the
 application which created the DTF file. (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
* wraps calls to \c dtf_query_application().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Info::queryAppVersion(string& version);
 * @brief not implemented
 *
 * @param version - reference to a string holding the version of that
 application which created the DTF file. (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
* wraps calls to \c dtf_query_appversion().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Info::queryCreateTime(time_t& created);
 * @brief get creation time of DTF file as time_t value
 *
 * @param created - reference to UNIX time_t value indicating when the file
 was created. (output)
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_cretime().
*/

/** @fn bool DTF_Lib::Info::queryDtfVersion(string& version);
 * @brief not implemented
 *
 * @param version - reference to string containing the version of the currently
used DTF library. (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
* wraps calls to \c dtf_query_dtf_version().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Info::queryFileVersion(string& version);
 * @brief gets version of DTF library with which the DTF file was created
 *
 * @param version - reference to string holding the DTF version of the file.
 (output)
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_file_version().
*/

/** @fn bool DTF_Lib::Info::queryModTime(time_t& modified);
 * @brief get modification time of the DTF file
 *
 * @param modified - reference to UNIX time_t value holding the last
 modification time of the file. (output)
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_modtime().
*/

/** @fn bool DTF_Lib::Info::queryOrigin(string& origin);
 * @brief get file origin of the DTF file
 *
 * @param origin - reference to a string holding the file origin. (output)
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_origin().
 */

/** @fn bool DTF_Lib::Info::queryScaling(double& scaling);
 * @brief get scaling factor of the DTF file
 *
 * @param scaling - reference to double holding the scaling factor of the
 file. (output)
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_scaling_d() and \c dtf_query_scaling_s().
*/

/** @fn bool DTF_Lib::Info::queryTitle(string& title);
 * @brief get title of DTF file
 *
 * @param title - reference to a string holding the file title. (output)
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_title().
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_INFO_H_
#define __DTF_LIB_INFO_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibInfo;

class Info : public LibObject
{
    friend class ClassInfo_DTFLibInfo;
    // Associations
    // Attributes
    // Operations
private:
    Info();
    Info(string className, int objectID);

    static ClassInfo_DTFLibInfo classInfo;

public:
    virtual ~Info();

    bool queryApplication(string &application);

    bool queryAppVersion(string &version);

    bool queryCreateTime(time_t &created);

    bool queryDtfVersion(string &version);

    bool queryFileVersion(string &version);

    bool queryModTime(time_t &modified);

    bool queryOrigin(string &origin);

    bool queryScaling(double &scaling);

    bool queryTitle(string &title);
};

CLASSINFO(ClassInfo_DTFLibInfo, Info);
};
#endif

/** EOC */
