/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/bcvc.h
 * @brief contains definition of class DTF_Lib::BcVc.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibBcVc
 * @brief used to register class DTF_Lib::BcVc at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::BcVc and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibBcVc to create new objects of type DTF_Lib::BcVc.
 */

/** @class DTF_Lib::BcVc
 * @brief access boundary condition to volume condition connectivity.
 *
 * Contains access functions for boundary condition to volume condition
 * connectivity.
 *
 * @see DTF_Lib::BcRecords.
 *
 * @see DTF_Lib::VcRecords.
 */

/** @fn DTF_Lib::BcVc::BcVc();
 * @brief default constructor
 *
 * \b Description:
 *
 * Initializes new objects. Calls default constructor of DTF_Lib::LibObject.
 */

/** @fn DTF_Lib::BcVc::BcVc( string className, int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param className - name of the class
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object. className is used
 * to hold statistics about objects of this class in statistic manager.
 */

/** @fn bool DTF_Lib::BcVc::~BcVc();
 * @brief default destructor
 *
 * \b Description:
 *
 * Destroys objects of type DTF_Lib::BcVc and tells statistic manager that
 * object has been deleted.
 */

/** @fn bool DTF_Lib::BcVc::queryBFnum( int simNum,
int zoneNum,
int faceIndex,
int& bfNum);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceIndex - index of the face in the global face array
 *
 * @param bfNum - boundary face number corresponding to supplied global face
index (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_query_bfnum_by_fnum().
*
* @attention You'll have to implement this function if you intend to use it. A
* warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcVc::queryIFnum( int simNum,
int zoneNum,
int faceIndex,
int& ifNum);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceIndex - index of the face in global face array
 *
 * @param ifNum - interface face number corresponding to supplied global face
index (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_query_ifnum_by_fnum().
*
* @attention You'll have to implement this function if you intend to use it. A
* warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcVc::queryBF2BCR( int simNum,
int zoneNum,
vector<int>& bfacesOfType);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param bfacesOfType - vector with number of boundary faces of each type.
This vector is \c DTF_NFACETYPES long and filled with the number of boundary
faces for each DTF face type. (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_query_bf2bcr().
*
* @attention You'll have to implement this function if you intend to use it. A
* warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcVc::queryXF2BCR( int simNum,
int zoneNum,
vector<int>& xfacesOfType);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param xfacesOfType - vector containing number of interface faces of each
 type. size = ? (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_query_xf2bcr().
*
* @attention You'll have to implement this function if you intend to use it. A
* warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcVc::readBF2BCR ( int simNum,
         int zoneNum,
         int faceNum,
         vector<int>& bf2f,
         vector<int>& bf2r );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceNum - boundary face number ( < 1 means all )
 *
* @param bf2f - vector with boundary face to global face connectivity (output)
* @param bf2r - vector with boundary face to BC record number connectivity (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_bf2bcr().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcVc::readBF2NBCR ( int simNum,
          int zoneNum,
          int faceNum,
          vector<int>& bf2n,
          vector<int>& bf2r );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceNum - boundary face number ( < 1 means all )
 *
* @param bf2n - vector with boundary -> node connectivity (output)
* @param bf2r - vector with boundary face -> BC record number connectivity (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_bf2nbcr().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcVc::readXF2BCR ( int simNum,
         int zoneNum,
         int faceNum,
         vector<int>& xf2f,
         vector<int>& xf2r );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceNum - interface face number ( < 1 means all )
 *
* @param xf2f - vector with interface face -> global face connectivity (output)
* @param xf2r - vector with interface face -> BC record number connectivity
(output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_read_xf2bcr().
*
* @attention You'll have to implement this function if you intend to use it. A
* warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::BcVc::readXF2NBCR ( int simNum,
          int zoneNum,
          int faceNum,
          vector<int>& xf2n,
          vector<int>& xf2r );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceNum - interface face number ( <1 means all )
 *
* @param xf2n - interface face -> node connectivity (output)
* @param xf2r - interface face -> BC record number connectivity (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_xf2nbcr().
 *
 * @attention You'll have to implement this function if you intend to use it. A
 * warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_BCVC_H_
#define __DTF_LIB_BCVC_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibBcVc;

class BcVc : public LibObject
{
    friend class ClassInfo_DTFLibBcVc;

    // Associations
    // Attributes
    // Operations
protected:
    BcVc();
    BcVc(string className, int objectID);

    static ClassInfo_DTFLibBcVc classInfo;

public:
    virtual ~BcVc();

    bool queryBFnum(int simNum,
                    int zoneNum,
                    int faceIndex,
                    int &bfNum);

    bool queryIFnum(int simNum,
                    int zoneNum,
                    int faceIndex,
                    int &ifNum);

    bool queryBF2BCR(int simNum,
                     int zoneNum,
                     vector<int> &bfacesOfType);

    bool queryXF2BCR(int simNum,
                     int zoneNum,
                     vector<int> &xfacesOfType);

    bool readBF2BCR(int simNum,
                    int zoneNum,
                    int faceNum,
                    vector<int> &bf2f,
                    vector<int> &bf2r);

    bool readBF2NBCR(int simNum,
                     int zoneNum,
                     int faceNum,
                     vector<int> &bf2n,
                     vector<int> &bf2r);

    bool readXF2BCR(int simNum,
                    int zoneNum,
                    int faceNum,
                    vector<int> &xf2f,
                    vector<int> &xf2r);

    bool readXF2NBCR(int simNum,
                     int zoneNum,
                     int faceNum,
                     vector<int> &xf2n,
                     vector<int> &xf2r);
};

CLASSINFO(ClassInfo_DTFLibBcVc, BcVc);
};
#endif

/** EOC */
