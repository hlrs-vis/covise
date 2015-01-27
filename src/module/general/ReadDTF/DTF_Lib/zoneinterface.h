/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/zoneinterface.h
 * @brief contains definition of class DTF_Lib::ZoneInterface
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibZoneInterface
 * @brief used to register class DTF_Lib::ZoneInterface at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::ZoneInterface and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibZoneInterface to create new objects of type DTF_Lib::ZoneInterface.
 */

/** @class DTF_Lib::ZoneInterface::
 * @brief contains functions to access informations about zonal interfaces.
 */

/** @fn DTF_Lib::ZoneInterface::ZoneInterface();
 * @brief default constructor
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject for initialization.
 */

/** @fn DTF_Lib::ZoneInterface::ZoneInterface(int* fileHandle);
 * @brief initializes new objects with given file handle
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject with given file handle as argument.
 */

/** @fn virtual DTF_Lib::ZoneInterface::~ZoneInterface();
 * @brief destructor
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::ZoneInterface::queryNumZI( int simNum,
         int& numZI );
 * @brief not implemented
 *
 * @param simNum - simulation number
 *
 * @param numZI - number of zonal interfaces for simulation (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
* \b Description:
 *
 * wraps calls to \c dtf_query_nzi().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneInterface::queryNumZIforZone ( int simNum,
                int zoneNum,
                int& numZI );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numZI - number of zonal interfaces for zone (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_nzi_zone().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneInterface::queryZI ( int simNum,
           int ziNum,
           int& leftZone,
           int& rightZone,
           int& numFaces );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param ziNum - zonal interface number
 *
 * @param leftZone - zone num to the left of the zonal interface (output)
* @param rightZone - zone num to the right of the zonal interface (output)
* @param numFaces - number of faces at the zonal interface (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_zi().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneInterface::queryZIforZone ( int simNum,
             int zoneNum,
             int& ziNum,
             int& leftZone,
             int& rightZone,
             int& numFaces );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
* @param ziNum - zonal interface number (output)
* @param leftZone - zone number left of this zone (output)
* @param rightZone - zone number right of this zone (output)
* @param numFaces - number of faces at this zonal interface (output)
*
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_zi_zone().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneInterface::readZI ( int simNum,
          int ziNum,
          vector<int>& facenums_l,
          vector<int>& facenums_r );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param ziNum - zonal interface number
 *
 * @param facenums_l - list of global face numbers on the left side of this
 * zonal interface (output)
* @param facenums_r - list of global face numbers on the left side of this
zonal interface (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_read_zi().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::ZoneInterface::readZIforZone ( int simNum,
            int zoneNum,
            int& ziNum,
            vector<int>& facenum_l,
            vector<int>& facenum_r );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param ziNum - zonal interface number (output)
* @param facenum_l - vector of facenumbers left of this zone (output)
* @param facenum_r - vector of facenumbers right of this zone (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_zi_zone().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef ZONEINTERFACE_H
#define ZONEINTERFACE_H

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibZoneInterface;

class ZoneInterface : public LibObject
{
    friend class ClassInfo_DTFLibZoneInterface;

    ZoneInterface();
    ZoneInterface(string className, int objectID);

    static ClassInfo_DTFLibZoneInterface classInfo;

public:
    virtual ~ZoneInterface();

    bool queryNumZI(int simNum,
                    int &numZI);

    bool queryNumZIforZone(int simNum,
                           int zoneNum,
                           int &numZI);

    bool queryZI(int simNum,
                 int ziNum,
                 int &leftZone,
                 int &rightZone,
                 int &numFaces);

    bool queryZIforZone(int simNum,
                        int zoneNum,
                        int &ziNum,
                        int &leftZone,
                        int &rightZone,
                        int &numFaces);

    bool readZI(int simNum,
                int ziNum,
                vector<int> &facenums_l,
                vector<int> &facenums_r);

    bool readZIforZone(int simNum,
                       int zoneNum,
                       int &ziNum,
                       vector<int> &facenum_l,
                       vector<int> &facenum_r);
};

CLASSINFO(ClassInfo_DTFLibZoneInterface, ZoneInterface);
};
#endif

/** EOC */
