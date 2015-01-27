/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/polyzonedata.h
 * @brief contains definition of class DTF_Lib::PolyZoneData
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibPolyZoneData
 * @brief used to register class DTF_Lib::PolyZoneData at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::PolyZoneData and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibPolyZoneData to create new objects of type DTF_Lib::PolyZoneData.
 */

/** @class DTF_Lib::PolyZoneData
 * @brief encapsulates poly zone data requested and returned by
 PolyZone::readSizes
 */

/** @fn DTF_Lib::PolyZoneData::PolyZoneData();
 * @brief default constructor
 *
 * \b Description:
 *
 * Sets all attributes to 0
 */

/** @fn DTF_Lib::PolyZoneData::PolyZoneData(map<string,int> values);
 * @brief initializes object with contents of map.
 *
 * \b Description:
 *
 * The map must be filled with following map keys:
 * - "numFaces"
 * - "numBFaces"
 * - "numXFaces"
 * - "numCells"
 * - "lengthF2N"
 * - "lengthF2C"
 * - "lengthC2N"
 * - "lengthC2F"
 */

/** @fn DTF_Lib::PolyZoneData::~PolyZoneData();
 * @brief destructor
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn	bool DTF_Lib::PolyZoneData::fill(map<string,int> values, string name);
 * @brief searches map for key 'name' and writes results to values.
 *
 * @param values - map with key, value pairs
 * @param name - name to search for in map
 *
 * @return \c false on error, \c true on success.
 */

/** @fn bool DTF_Lib::PolyZoneData::getValue(string key, int& value);
 * @brief get value from map with given key
 *
 * @param key - key to search for in map
 *
 * @param value - found value (output)
 *
 * @return \c false on erro, \c true on success.
 */

/** @fn bool DTF_Lib::PolyZoneData::setValue(string key, int value);
 * @brief get value from map with given key
 *
 * @param key - key to set for in map
 * @param value - value for key in map
 *
 * @return \c false on erro, \c true on success.
 */

/** @var map<string,int> DTF_Lib::PolyZoneData::values;
 * @brief sizing informations of poly zone
 *
 * \b Description:
 *
 * Contains some sizing specific data for a poly zone.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_POLYZONEDATA_H_
#define __DTF_LIB_POLYZONEDATA_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibPolyZoneData;

class PolyZoneData : public Tools::BaseObject
{
    friend class ClassInfo_DTFLibPolyZoneData;

private:
    map<string, int> values;
    map<string, int>::iterator valIterator;

    PolyZoneData();
    PolyZoneData(string className, int objectID);

private:
    bool fill(map<string, int> values, string name);

    static ClassInfo_DTFLibPolyZoneData classInfo;

public:
    virtual ~PolyZoneData();
    virtual void clear();

    bool getValue(string key, int &value);
    bool setValue(string key, int value);
};

CLASSINFO(ClassInfo_DTFLibPolyZoneData, PolyZoneData);
};
#endif

/** EOC */
