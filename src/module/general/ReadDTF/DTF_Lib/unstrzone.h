/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/unstrzone.h
 * @brief contains definition of class DTF_Lib::UnstrZone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibUnstrZone
 * @brief used to register class DTF_Lib::UnstrZone at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::UnstrZone and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibUnstrZone to create new objects of type DTF_Lib::UnstrZone.
 */

/** @class DTF_Lib::UnstrZone
 * @brief contains access functions for unstructured zones.
 */

/** @fn DTF_Lib::UnstrZone::UnstrZone();
 * @brief default constructor
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject to do the actual init process.
 */

/** @fn DTF_Lib::UnstrZone::UnstrZone( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn DTF_Lib::UnstrZone::~UnstrZone()
 * @brief default destructor
 *
 * \b Description:
 *
 * Called when objects are destroyed
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_UNSTRZONE_H_
#define __DTF_LIB_UNSTRZONE_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibUnstrZone;

class UnstrZone : public LibObject
{
    friend class ClassInfo_DTFLibUnstrZone;

    UnstrZone();
    UnstrZone(string className, int objectID);

    static ClassInfo_DTFLibUnstrZone classInfo;

public:
    virtual ~UnstrZone();
};

CLASSINFO(ClassInfo_DTFLibUnstrZone, UnstrZone);
};
#endif

/** EOC */
