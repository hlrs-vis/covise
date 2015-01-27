/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/unstrzone.cpp
 * @brief contains definition of methods for class DTF_Lib::UnstrZone
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "unstrzone.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibUnstrZone, UnstrZone, "DTF_Lib::UnstrZone", 1);

UnstrZone::UnstrZone()
    : LibObject(){};

UnstrZone::UnstrZone(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

UnstrZone::~UnstrZone()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

/** EOC */
