/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/file.h
 * @brief contains definition of class DTF_Lib::File
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibFile
 * @brief used to register class DTF_Lib::File at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::File and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibFile to create new objects of type DTF_Lib::File.
 */

/** @class DTF_Lib::File
 * @brief contains some functions to open and close DTF files.
 *
 * @note The created file handle must be propagated to all classes in DTF_Lib.
 */

/** @fn DTF_Lib::File::File();
 * @brief default constructor.
 *
 * \b Description:
 *
 * calls default constructor of class DTF_Lib::LibObject.
 */

/** @fn DTF_Lib::File::File( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::File::~File();
 * @brief default destructor.
 *
 * \b Description:
 *
 * called when objects of DTF_Lib::File are destroyed.
 */

/** @fn int* DTF_Lib::File::getFileHandle();
 * @brief get file handle associated to an open DTF file.
 *
 * @param fileHandle - returned file handle (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * returns the private member fileHandle to the caller.
 */

/** @fn bool DTF_Lib::File::open(string fileName);
 * @brief open DTF file.
 *
 * @param fileName - name of the file which is to open.
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_open_file().
 */

/** @fn bool DTF_Lib::File::close()
 * @brief close currently open DTF file.
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_close_file().
 *
 */

/** @fn bool DTF_Lib::File::queryNumSims(int& numSims)
 * @brief get number of simulations in DTF file
 *
 * @param numSims - number of simulations in file (output)
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_nsims().
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_FILE_H_
#define __DTF_LIB_FILE_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibFile;

class File : public LibObject
{
    friend class ClassInfo_DTFLibFile;

private:
    File();
    File(string className, int objectID);

    static ClassInfo_DTFLibFile classInfo;

public:
    virtual ~File();

    virtual bool setFileName(string fileName);
    virtual int getFileHandle();

    bool open(string fileName);
    bool close();

    bool queryNumSims(int &numSims);
};

CLASSINFO(ClassInfo_DTFLibFile, File);
};
#endif

/** EOC */
