/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/misc.h
 * @brief contains definition of class DTF_Lib::Misc
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibMisc
 * @brief used to register class DTF_Lib::Misc at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::Misc and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibMisc to create new objects of type DTF_Lib::Misc.
 */

/** @class DTF_Lib::Misc
 * @brief empty useless class
 *
 * \b Description:
 *
 * this class is entirely useless and provided as easter-egg.
 */

/** @fn DTF_Lib::Misc::Misc()
 * @brief default constructor
 *
 * \b Description:
 *
 * Initializes new objects.
 */

/** @fn virtual DTF_Lib::Misc::~Misc()
 * @brief default destructor
 *
 * \b Description:
 *
 * Called when object is destroyed
 */

/** @fn void DTF_Lib::Misc::mostImportantFunction()
 * @brief really important function
 *
 * \b Description:
 *
 * Just kidding. Is easter-eggy ;)
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_MISC_H_
#define __DTF_LIB_MISC_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibMisc;

class Misc : public LibObject
{
    friend class ClassInfo_DTFLibMisc;

private:
    Misc();
    Misc(string className, int objectID);

    static ClassInfo_DTFLibMisc classInfo;

public:
    virtual ~Misc();

    void mostImportantFunction();
};

CLASSINFO(ClassInfo_DTFLibMisc, Misc);
};
#endif

/** EOC */
