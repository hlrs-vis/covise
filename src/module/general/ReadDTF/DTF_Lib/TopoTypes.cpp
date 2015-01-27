/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** file DTF_Lib/TopoTypes.cpp
 * @brief contains implementation of methods of class DTF_Lib::TopoTypes
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.10.2003
 * created
 */

#include "TopoTypes.h"

using namespace DTF_Lib;

TopoTypes::TopoTypes()
    : Tools::EnumTypes()
{
    names.insert(pair<int, string>(0, "None"));
    names.insert(pair<int, string>(1, "Node"));
    names.insert(pair<int, string>(2, "Edge"));
    names.insert(pair<int, string>(3, "Face"));
    names.insert(pair<int, string>(4, "Cell"));
}

TopoTypes::~TopoTypes()
{
}
