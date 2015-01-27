/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF_Lib/DataTypes.h
 * @brief contains declaration of class DTF_Lib::DataTypes
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.10.2003
 * created
 */

#ifndef __DTF_DATATYPES_H_
#define __DTF_DATATYPES_H_

#include "../Tools/EnumTypes.h"

namespace DTF_Lib
{
class DataTypes : public Tools::EnumTypes
{
public:
    friend class Tools::Singleton<DataTypes>::InstanceHolder;

private:
    DataTypes();

public:
    virtual ~DataTypes();
};
};
#endif
