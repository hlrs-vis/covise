/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/volume.h
 * @brief contains definition of class DTF_Lib::Volume
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibVolume
 * @brief used to register class DTF_Lib::Volume at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::Volume and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibVolume to create new objects of type DTF_Lib::Volume.
 */

/** @class DTF_Lib::Volume
 * @brief contains functions related to volume conditions
 */

/** @fn DTF_Lib::Volume::Volume();
 * @brief default constructor
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject for initialization.
 */

/** @fn DTF_Lib::Volume::Volume( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::Volume::~Volume();
 * @brief destructor.
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::Volume::queryCondition ( int simNum,
             int zoneNum,
             int conditionNum,
             int& groupNum,
             int& recordNum );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param conditionNum - condition number
 *
* @param groupNum - volume condition group number (output)
* @param recordNum - volume condition record number (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_volume_condition().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Volume::queryNumConditions ( int simNum,
            int zoneNum,
            int& numConditions );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numConditions - number of volume conditions in given zone (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_nvolume_conditions().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_VOLUME_H_
#define __DTF_LIB_VOLUME_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibVolume;

class Volume : public LibObject
{
    friend class ClassInfo_DTFLibVolume;

    Volume();
    Volume(string className, int objectID);

    static ClassInfo_DTFLibVolume classInfo;

public:
    virtual ~Volume();

    bool queryCondition(int simNum,
                        int zoneNum,
                        int conditionNum,
                        int &groupNum,
                        int &recordNum);

    bool queryNumConditions(int simNum,
                            int zoneNum,
                            int &numConditions);
};

CLASSINFO(ClassInfo_DTFLibVolume, Volume);
};
#endif

/** EOC */
