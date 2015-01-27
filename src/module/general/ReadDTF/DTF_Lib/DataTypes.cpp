/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF_Lib/DataTypes.cpp
 * @brief contains implementation of methods of class DTF_Lib::CellTypes
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.10.2003
 * created
 */

#include "DataTypes.h"

using namespace DTF_Lib;

DataTypes::DataTypes()
    : Tools::EnumTypes()
{
    names.insert(pair<int, string>(1, "dtf_int"));
    names.insert(pair<int, string>(2, "dtf_double"));
    names.insert(pair<int, string>(3, "dtf_single"));
    names.insert(pair<int, string>(4, "dtf_string"));
}

DataTypes::~DataTypes()
{
}
