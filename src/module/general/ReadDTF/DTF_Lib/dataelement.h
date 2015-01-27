/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/dataelement.h
 * @brief contains definition of class DTF_Lib::DataElement
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 10.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope.
 */

/** @class DTF_Lib::ClassInfo_DTFLibDataElement
 * @brief used to register class DTF_Lib::DataElement at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::DataElement and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibDataElement to create new objects of type DTF_Lib::DataElement.
 */

/** @class DTF_Lib::DataElement
 * @brief encapsulates zone and sim data informations
 *
 * This class is used by DTF_Lib::VirtualZone, DTF_Lib::ZoneData, and
 * DTF_Lib::SimData as return type when information about zone data is queried.
 */

/** @fn DTF_Lib::DataElement::DataElement();
 * @brief default constructor
 *
 * \b Description:
 *
 * Is private to avoid instantiation of empty objects.
 */

/** @fn DTF_Lib::DataElement::DataElement(string name,
          int numElements,
          int dataType,
          string units,
          int topoType);
 * @brief fills object with values
 *
 * @param name - name of the data element
 * @param numElements - number of elements in data element
 * @param dataType - type of the data contained in data element
 * @param units - string describing the data element's units
* @param topoType - topological type of the data element
*
* \b Description:
*
 * To avoid a lot of set-functions new DataElements are created with this
 * constructor and can never be changed.
 */

/** @fn DTF_Lib::DataElement::~DataElement();
 * @brief destructor
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn	string DTF_Lib::DataElement::getName();
 * @brief get name of data element
 *
 * @return name of the data element.
 */

/** @fn int DTF_Lib::DataElement::getNumElements();
 * @brief get number of elements
 *
 * @return number of elements in data element
 */

/** @fn int DTF_Lib::DataElement::getDataType();
 * @brief get datatype of data element
 *
 * @return type of data element
 */

/** @fn string DTF_Lib::DataElement::getUnits();
 * @brief get units of the data element
 *
 * @return string describing the units of the data element
 */

/** @fn int DTF_Lib::DataElement::getTopoType();
 * @brief get topo type of the data element
 *
 * @return topological type of the data element
 */

/** @var DTF_Lib::DataElement::name
 * @brief name of the data element
 */

/** @var DTF_Lib::DataElement::numElements
 * @brief number of elements of the data element
 */

/** @var DTF_Lib::DataElement::dataType
 * @brief type of the data contained in data element
 */

/** @var DTF_Lib::DataElement::units
 * @brief string describing the units of the data element
 */

/** @var DTF_Lib::DataElement::topoType
 * @brief topological type of the data element
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_DATAELEMENT_H_
#define __DTF_LIB_DATAELEMENT_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibDataElement;

class DataElement : public Tools::BaseObject
{
    friend class ClassInfo_DTFLibDataElement;

private:
    string name;
    int numElements;
    int dataType;
    string units;
    int topoType;

    DataElement();
    DataElement(string className, int objectID);

    static ClassInfo_DTFLibDataElement classInfo;

public:
    virtual ~DataElement();

    bool setValues(string name,
                   int numElements,
                   int dataType,
                   string units,
                   int topoType);

    string getName();
    int getNumElements();
    int getDataType();
    string getUnits();
    int getTopoType();

    virtual void clear();
};

CLASSINFO(ClassInfo_DTFLibDataElement, DataElement);
};
#endif

/** EOC */
