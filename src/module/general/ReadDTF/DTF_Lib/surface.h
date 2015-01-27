/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/surface.h
 * @brief contains definition of class DTF_Lib::Surface
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibSurface
 * @brief used to register class DTF_Lib::Surface at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::Surface and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibSurface to create new objects of type DTF_Lib::Surface.
 */

/** @class DTF_Lib::Surface
 * @brief contains access functions to informations about surface conditions
 and faces
 */

/** @fn DTF_Lib::Surface::Surface();
 * @brief default constructor.
 *
 * \b Description:
 *
 * Calls constructor of DTF_Lib::LibObject to do the actual initialization.
 */

/** @fn DTF_Lib::Surface::Surface( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::Surface::~Surface();
 * @brief default destructor.
 *
 * \b Description:
 *
 * Called when objects are destroyed.
 */

/** @fn bool DTF_Lib::Surface::queryNumConditions(int simNum,
            int zoneNum,
            int& numConditions );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numConditions - number of surface conditions (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_nsurface_conditions().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Surface::queryCondition( int simNum,
             int zoneNum,
             int condNum,
             int& groupNum,
             int& recordNum );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param condNum - surface condition number
 *
* @param groupNum - surface condition group number (output)
* @param recordNum - surface condition record number (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_surface_condition().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Surface::queryNumGroups( int simNum,
             int zoneNum,
             int& numGroups );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numGroups - number of face groups (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_nface_groups().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Surface::queryGroup( int simNum,
         int zoneNum,
         int groupNum,
         int& key );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param groupNum - face group number
 *
 * @param key - key (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_face_group().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Surface::readGroup( int simNum,
             int zoneNum,
             int groupNum,
             vector<int>& faces );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param groupNum - group number
 *
 * @param faces - vector containing face numbers (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_face_group().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Surface::queryFaces( int simNum,
         int zoneNum,
         vector<int>& numTypes,
         vector<int>& numKind );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numTypes - number of faces for each face type (output)
 * @param numKind - number of faces for each face kind (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_faces().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Surface::queryNumFaces( int simNum,
            int zoneNum,
            int& numFaces );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numFaces - number of faces in zone (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_nfaces().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Surface::queryFaceKind( int simNum,
            int zoneNum,
            int faceNum,
            int& facekind );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceNum - face number
 *
 * @param facekind - face kind for given face number (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_facekind().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_SURFACE_H_
#define __DTF_LIB_SURFACE_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibSurface;

class Surface : public LibObject
{
    friend class ClassInfo_DTFLibSurface;

    Surface();
    Surface(string className, int objectID);

    static ClassInfo_DTFLibSurface classInfo;

public:
    virtual ~Surface();

    bool queryNumConditions(int simNum,
                            int zoneNum,
                            int &numConditions);

    bool queryCondition(int simNum,
                        int zoneNum,
                        int condNum,
                        int &groupNum,
                        int &recordNum);

    bool queryNumGroups(int simNum,
                        int zoneNum,
                        int &numGroups);

    bool queryGroup(int simNum,
                    int zoneNum,
                    int groupNum,
                    int &key);

    bool readGroup(int simNum,
                   int zoneNum,
                   int groupNum,
                   vector<int> &faces);

    bool queryFaces(int simNum,
                    int zoneNum,
                    vector<int> &numTypes,
                    vector<int> &numKind);

    bool queryNumFaces(int simNum,
                       int zoneNum,
                       int &numFaces);

    bool queryFaceKind(int simNum,
                       int zoneNum,
                       int faceNum,
                       int &facekind);
};

CLASSINFO(ClassInfo_DTFLibSurface, Surface);
};
#endif

/** EOC */
