/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/libif.h
 * @brief contains definition of class DTF_Lib::LibIF
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibLibIF
 * @brief used to register class DTF_Lib::LibIF at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::LibIF and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibLibIF to create new objects of type DTF_Lib::LibIF.
 */

/** @class DTF_Lib::LibIF
 * @brief interface to DTF lib from CFDRC.
 *
 * \b Description:
 *
 * Encapsulates access to DTF lib functions which are ugly in some places.
 *
 * They're also a pain in the ass ;) So this interface should enable sorted
 * access to interface functions.
 *
 * More docu follows....
 *
 */

/** @fn DTF_Lib::LibIF::LibIF();
 * @brief default constructor
 *
 * \b Description:
 *
 * Initializes interface to DTF library provided by CFDRC.
 */

/** @fn DTF_Lib::LibIF::LibIF(int objectID);
 * @brief initializes new objects with given ID
 *
 * @param objectID - unique identifier for the object
 *
 * \b Description:
 *
 * New objects are created by the class manager which also supplies a unique
 * object identifier.
 */

/** @fn DTF_Lib::LibIF::~LibIF();
 * @brief default destructor
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * Called when singleton object of DTF_Lib::LibIF is destroyed. Closes library
 * interface.
 */

/** @fn bool DTF_Lib::LibIF::setFileName(string fileName);
 * @brief set DTF file name
 *
 * @param fileName - path and name of the DTF file
 *
 * @return true if file could be opened. false on error.
 *
 * \b Description:
 *
 * changes filename for the DTF file used by library interface and propagates
 * the changed file handle to all classes contained in library interface.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_LIBIF_H_
#define __DTF_LIB_LIBIF_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibLibIF;

// singleton
class LibIF : public Tools::BaseObject
{
private:
    string fileName; /**< contains file name of DTF file */
    // Operations

    LibIF();
    LibIF(string className, int objectID);

    static ClassInfo_DTFLibLibIF classInfo;

public:
    virtual ~LibIF();
    bool setFileName(string fileName);
    Tools::BaseObject *operator()(string className);
    virtual bool init();

    friend class ClassInfo_DTFLibLibIF;
};

CLASSINFO(ClassInfo_DTFLibLibIF, LibIF);
};
#endif

/** EOC */
