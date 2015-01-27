/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/polyzone.h
 * @brief contains definition of class DTF_Lib::PolyZone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibPolyZone
 * @brief used to register class DTF_Lib::PolyZone at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::PolyZone and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibPolyZone to create new objects of type DTF_Lib::PolyZone.
 */

/** @class DTF_Lib::PolyZone
 * @brief contains access functions for informations about polyhedral zones.
 *
 * \b Description:
 *
 * Polyhedral grid zones are used to store unstructured
 * grids which may have faces with an arbitrary number of
 * nodes, and cells with an arbitrary number of faces.
 */

/** @fn DTF_Lib::PolyZone::PolyZone();
 * @brief default constructor.
 *
 * \b Description:
 *
 * Calls constructor of base class DTF_Lib::LibObject to initialize new objects.
 */

/** @fn DTF_Lib::PolyZone::PolyZone( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::PolyZone::~PolyZone();
 * @brief default destructor.
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::PolyZone::queryIsSorted(int simNum,
            int zoneNum,
            bool& result );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param result - indicates if zone is sorted or not (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_ispoly().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::PolyZone::querySizes(int simNum,
         int zoneNum,
         PolyZoneData& pzData);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param pzData - object of type DTF_Lib::PolyZoneData which contains the
 * queried sizes.
 *
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_poly_sizes().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_POLYZONE_H_
#define __DTF_LIB_POLYZONE_H_

#include "polyzonedata.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibPolyZone;

class PolyZone : public LibObject
{
    friend class ClassInfo_DTFLibPolyZone;

    // Associations
    // Attributes
    // Operations
private:
    PolyZone();
    PolyZone(string className, int objectID);

    static ClassInfo_DTFLibPolyZone classInfo;

public:
    virtual ~PolyZone();

    bool queryIsSorted(int simNum,
                       int zoneNum,
                       bool &result);

    bool querySizes(int simNum,
                    int zoneNum,
                    PolyZoneData &pzData);
};

CLASSINFO(ClassInfo_DTFLibPolyZone, PolyZone);
};
#endif

/** EOC */
