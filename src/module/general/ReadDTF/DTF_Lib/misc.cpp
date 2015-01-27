/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/misc.cpp
 * @brief contains definition of methods for class DTF_Lib::Misc
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "misc.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibMisc, Misc, "DTF_Lib::Misc", 1);

Misc::Misc(){};
Misc::Misc(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

Misc::~Misc()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

void Misc::mostImportantFunction()
{
    cout << "this function is really important and needed if you want the library interface to work!" << endl;
    cout << "okay, i'm just kidding. this function is really only an easter-egg" << endl;
}

/** EOC */
