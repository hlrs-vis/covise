/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/libobject.h
 * @brief contains definition of class DTF_Lib::LibObject.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 *
 * LibObject is the base class for all classes in namespace DTF_Lib.
 */

/** @class DTF_Lib::LibObject
 * @brief base class for all classes in namespace DTF_Lib.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 *
 * Defines some basic attributes and functions used by all classes in its
 * namespace
 */

/** @fn DTF_Lib::LibObject::LibObject();
 * @brief default constructor
 *
 * \b Description:
 *
 * Initializes new objects of DTF_Lib::LibObject.
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn DTF_Lib::LibObject::LibObject( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn DTF_Lib::LibObject::~LibObject();
 * @brief destructor.
 *
 * \b Description:
 *
 * Called when objects are removed from memory. Calls \a clear() to clean-up
 * the memory occupied by the object before it is removed.
 */

/** @fn virtual bool DTF_Lib::LibObject::setFileName (string fileName);
 * @brief set name of the DTF file.
 *
 * @param fileName - path to the DTF file which should be used by the library
 * interface object
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * Sets the name of the DTF file which should be used inside the library
 * interface.
 */

/** @fn virtual int DTF_Lib::LibObject::getFileHandle();
 * @brief get handle pointing to an open DTF file.
 *
 * @return handle to an open DTF file. If no DTF file is open, then the
 * return value is unspecified and the return value of
 * DTF_Lib::LibIF::setFileName() should be checked.
 *
 * \b Description:
 *
 * Returns the handle to an open DTF file. File handles should never be set
 * explicitly. It is recommended to use the function
 * DTF_Lib::LibIF::setFileName() to avoid use of multiple DTF files in the
 * library interface.
 */

/** @fn virtual bool DTF_Lib::LibObject::setFileHandle(int handle);
 * @brief set handle to open DTF file.
 *
 * @param handle - valid file handle to an open DTF file.
 *
 * @return \c false on error, \c true on success.
 *
 * \b Description:
 *
 * Sets the member variable fileHandle to given argument. The file handle
 * is used when accessing the DTF library of CFDRC.
 */

/** @fn bool DTF_Lib::LibObject::implementMe();
 * @brief used by unimplemented functions of derived classes.
 *
 * @return always false, since this function indicates unimplemented functions.
 *
 * \b Description:
 *
 * Prints warning that the called function is not implemented yet.
 */

/** @var int* DTF_Lib::LibObject::fileHandle;
 * @brief handle to DTF file
 *
 * \b Description:
 *
 * Holds handle to an open DTF file.
 */

/** @var string DTF_Lib::LibObject::fileName;
 * @brief path to the DTF file
 *
 * \b Description:
 *
 * Contains the path to the DTF file. Please note that the only relevant DTF
 * filename is contained in the DTF_Lib::LibIF object.
 */

/** EOD */

/** BOC */

#ifndef DTF_LIB_LIBOBJECT_H
#define DTF_LIB_LIBOBJECT_H

#include "../Tools/classmanager.h"

using namespace std;

/** @namespace DTF_Lib
 * @brief contains interface classes used to access DTF library from CFDRC
 */
namespace DTF_Lib
{
class LibObject : public Tools::BaseObject
{

protected:
    int fileHandle;
    string fileName;

    LibObject();
    LibObject(string className, int objectID);

public:
    virtual ~LibObject();

    virtual bool setFileName(string fileName);
    virtual int getFileHandle();
    virtual bool setFileHandle(int handle);

protected:
    bool implementMe();
};
};
#endif

/** EOC */
