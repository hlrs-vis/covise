/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file EnumTypes.h
 * @brief Contains definition of singleton class Tools::EnumTypes.
 * @author Alexander Martinez <kubus3561@gmx.de>
 */

/** @class Tools::EnumTypes
 * @brief Singleton class used to ask for element types, names and length
 *
 * Derive and implement specific classes to hold specific informations about
 * element types.
 *
 * Derived classes should be thought of as wrappers for enum values defined by
 * DTF library from CFDRC.
 */

/** @fn Tools::EnumTypes::EnumTypes()
 * @brief default constructor.
 *
 * Initializes empty member maps.
 */

/** @fn Tools::EnumTypes::~EnumTypes()
 * @brief Default destructor
 *
 * Deletes member maps containing element types, names and length information
 */

/** @fn int Tools::EnumTypes::toString(int typeNumber, string& typeName)
 * @brief get name for given type number
 *
 * @param typeNumber - number of the type
 * @param typeName - string which holds the returned type name
 *
 * @return 1 on success, 0 on error. If type number was found then it returns
 * also the type name.
 *
 * Returns a string describing the given type number. Intended for debug
 * outputs.
 */

/** @fn int Tools::EnumTypes::toInt(
 string typeName,
 int& typeNumber)
 * @brief returns type number for given type name.
 *
 * @param typeName - name of the type
 * @param typeNumber - reference to an int holding the returned type number
 *
 * @return 1 on success, 0 on error
 *
 * Returns the type number associated with a given type name.
*/

#ifndef __TOOLS_ENUMTYPES_H_
#define __TOOLS_ENUMTYPES_H_

#include "Singleton.h"

using namespace std;

namespace Tools
{

class EnumTypes
{
    friend class Singleton<EnumTypes>::InstanceHolder;

protected:
    EnumTypes();

protected:
    map<int, string> names; ///< holds type number -> name information

public:
    virtual ~EnumTypes();

    int toString(int typeNumber,
                 string &typeName);

    int toInt(string typeName,
              int &typeNumber);
};
};
#endif
