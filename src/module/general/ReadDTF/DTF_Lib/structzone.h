/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/structzone.h
 * @brief contains definition of class DTF_Lib::StructZone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibStructZone
 * @brief used to register class DTF_Lib::StructZone at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::StructZone and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibStructZone to create new objects of type DTF_Lib::StructZone.
 */

/** @class DTF_Lib::StructZone
 * @brief contains access functions for informations related to structured
 zones
 */

/** @fn DTF_Lib::StructZone::StructZone();
 * @brief default constructor.
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject.
 */

/** @fn DTF_Lib::StructZone::StructZone( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::StructZone::~StructZone();
 * @brief default destructor.
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::StructZone::queryDims( int simNum,
             int zoneNum,
             vector<int>& dims );
 * @brief get dimensions of structured grid zone
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param dims - vector containing the dimensions of the grid (output)
 *
 * @return \c false on error, \c true on success.
*
* \b Description:
 *
 * wraps calls to \c dtf_query_dims().
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_STRUCTZONE_H_
#define __DTF_LIB_STRUCTZONE_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibStructZone;

class StructZone : public LibObject
{
    friend class ClassInfo_DTFLibStructZone;

    StructZone();
    StructZone(string className, int objectID);

    static ClassInfo_DTFLibStructZone classInfo;

public:
    virtual ~StructZone();

    bool queryDims(int simNum,
                   int zoneNum,
                   vector<int> &dims);

    virtual bool init();
    Tools::BaseObject *operator()(string className);
    virtual bool setFileHandle(int handle);
};

CLASSINFO(ClassInfo_DTFLibStructZone, StructZone);
};
#endif

/** EOC */
