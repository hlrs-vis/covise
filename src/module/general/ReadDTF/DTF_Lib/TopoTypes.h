/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file DTF_Lib/TopoTypes.h
 * @brief contains definition of class DTF_Lib::TopoTypes
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 09.10.2003
 * created
 */

#ifndef __DTF_TOPOTYPES_H_
#define __DTF_TOPOTYPES_H_

#include "../Tools/EnumTypes.h"

namespace DTF_Lib
{
class TopoTypes : public Tools::EnumTypes
{
public:
    friend class Tools::Singleton<TopoTypes>::InstanceHolder;

private:
    TopoTypes();

public:
    virtual ~TopoTypes();
};
};
#endif
