/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/CellTypes.h
 * @brief Contains declaration of class DTF_Lib::CellTypes
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.10.2003
 * created
 */

/** @class DTF_Lib::CellTypes
 * @brief encapsulated enum for cell types supported by CFD-DTF.
 *
 * Derived from Tools::EnumTypes which contains the basic access functions
 * needed to make the actual enum value -> name translations.
 */

/** @fn DTF_Lib::CellTypes::CellTypes();
 * @brief default constructor
 *
 * \b Description:
 *
 * Private
 */

/** EOD */

/** BOC */
#ifndef __DTF_CELLTYPES_H_
#define __DTF_CELLTYPES_H_

#include "../Tools/EnumTypes.h"

using namespace std;

namespace DTF_Lib
{
class CellTypes : public Tools::EnumTypes
{
    friend class Tools::Singleton<CellTypes>::InstanceHolder;

private:
    CellTypes();

    map<int, int> elementLength;

public:
    virtual ~CellTypes();
};
};
#endif

/** EOC */
