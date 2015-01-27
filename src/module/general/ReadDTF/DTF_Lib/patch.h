/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/patch.h
 * @brief contains definition of class DTF_Lib::Patch
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibPatch
 * @brief used to register class DTF_Lib::Patch at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::Patch and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibPatch to create new objects of type DTF_Lib::Patch.
 */

/** @class DTF_Lib::Patch
 * @brief contains access functions for data related to patches.
 *
 * \b Description:
 *
 * Patches associate structured grid faces with boundary condition data.
 * Each patch stores a range of indices
 *
 * - imin
 * - imax
 * - jmin
 * - jmax
 * - kmin
 * - kmax
 *
 * and an array of boundary condition record numbers. They associate each face
 * in the patch with a boundary condition record.
 */

/** @fn DTF_Lib::Patch::Patch();
 * @brief default constructor
 *
 * \b Description:
 *
 * Calls constructor of superclass DTF_Lib::LibObject.
 */

/** @fn DTF_Lib::Patch::Patch( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn DTF_Lib::Patch::~Patch();
 * @brief default destructor
 *
 * \b Description:
 *
 * Called when object is destroyed.
 */

/** @fn bool DTF_Lib::Patch::queryNumPatches( int simNum,
              int zoneNum,
              int& numPatches );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numPatches - number of patches in given simulation/zone (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_npatches().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Patch::queryPatch(int simNum,
         int zoneNum,
         int patchNum,
         map<string, vector<int> >& minMax);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param patchNum - patch number
 *
 * @param minMax - min max values as STL map; keys are:
* - "i" (first plane min-max values)
* - "j" (second plane min-max values)
* - "k" (third plane min-max values)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_patch().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Patch::readPatch(int simNum,
             int zoneNum,
             int patchNum,
             vector<int>& records );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param patchNum - patch number
 *
 * @param records - records attached to the plane (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_patch().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_PATCH_H_
#define __DTF_LIB_PATCH_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibPatch;

class Patch : public LibObject
{
    friend class ClassInfo_DTFLibPatch;

private:
    Patch();
    Patch(string className, int objectID);

    static ClassInfo_DTFLibPatch classInfo;

public:
    virtual ~Patch();

    bool queryNumPatches(int simNum,
                         int zoneNum,
                         int &numPatches);

    bool queryPatch(int simNum,
                    int zoneNum,
                    int patchNum,
                    map<string, vector<int> > &minMax);

    bool readPatch(int simNum,
                   int zoneNum,
                   int patchNum,
                   vector<int> &records);
};

CLASSINFO(ClassInfo_DTFLibPatch, Patch);
};
#endif

/** EOC */
