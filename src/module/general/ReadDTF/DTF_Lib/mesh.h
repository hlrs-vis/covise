/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/mesh.h
 * @brief contains definition of class DTF_Lib::Mesh
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope
 */

/** @class DTF_Lib::ClassInfo_DTFLibMesh
 * @brief used to register class DTF_Lib::Mesh at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::Mesh and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibMesh to create new objects of type DTF_Lib::Mesh.
 */

/** @class DTF_Lib::Mesh
 * @brief provides access functions related to mesh connectivity
 */

/** @fn DTF_Lib::Mesh::Mesh();
 * @brief default constructor.
 *
 * \b Description:
 *
 * calls default constructor of DTF_Lib::LibObject to do some basic
 * initializations
 */

/** @fn DTF_Lib::Mesh::Mesh( int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::Mesh::~Mesh();
 * @brief default destructor.
 *
 * \b Description:
 *
 * Called when objects of DTF_Lib::Mesh are destroyed.
 */

/** @fn bool DTF_Lib::Mesh::queryC2F(int simNum,
            int zoneNum,
            int cellNum,
            vector<int>& facesPerCell );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param cellNum - number of cell which is to ask for cell->face data
 ( <1 means all ).
 *
* @param facesPerCell - vector containing the number of faces for each cell
type in the given cell (output).
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_query_c2f().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Mesh::queryC2N(int simNum,
            int zoneNum,
            int cellNum,
            vector<int>& nodesPerCell );
 * @brief get size of cell2node connectivity array and number of nodes/cell
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param cellNum - cell number ( < 1 means all )
 *
 * @param nodesPerCell - vector containing the number of nodes per cell (output)
*
* @return \c false on error, \c true on success.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_c2n().
 */

/** @fn bool DTF_Lib::Mesh::queryC2Npos(int simNum,
          int zoneNum,
          int cellNum, int& offset );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param cellNum - cell number
 *
 * @param offset - integer offset to global cell->node array for given cell
number (output).
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_query_c2n_pos().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Mesh::queryF2C(int simNum,
            int zoneNum,
            int faceNum,
            int& numF2C );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceNum - face number ( <1 means all faces)
 *
 * @param numF2C - size of the face->cell connectivity array (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_f2c().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Mesh::queryF2N(int simNum,
            int zoneNum,
            int faceNum,
            vector<int>& nodesPerFace);
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceNum - face number ( <1 means all)
 *
 * @param nodesPerFace - vector holding the number of nodes per face (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_f2n().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Mesh::queryF2Npos(int simNum,
          int zoneNum,
          int faceNum,
          int& offset );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceNum - face number
 *
 * @param offset - integer offset to face->node array (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_f2n_pos().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Mesh::queryN2C(int simNum,
            int zoneNum,
            int nodeNum,
            vector<int>& cellsPerNode );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param nodeNum - node number ( <1 means all )
 *
 * @param cellsPerNode - vector containing the number of cells per node (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_n2c().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Mesh::readC2F(int simNum,
           int zoneNum,
           int cellNum,
           vector<int>& c2f );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param cellNum - cell number
 *
 * @param c2f - vector containing the cell->face connectivity array for the
given cell(s) (output).
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
* \b Description:
*
* wraps calls to \c dtf_read_c2f().
*
* @attention You'll have to implement this function if you intend to use it.
* A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Mesh::readC2N(int simNum,
           int zoneNum,
           int cellNum,
           vector<int>& c2n );
 * @brief get cell->node connectivity array for given sim/zone/cell
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param cellNum - cell number ( < 1 means all )
 *
 * @param c2n - vector containing the cell->node connectivity array for the
given cell(s) (output).
*
* @return \c false on error, \c true on success.
*
* \b Description:
*
* wraps calls to \c dtf_read_c2n().
*/

/** @fn bool DTF_Lib::Mesh::readF2C ( int simNum,
           int zoneNum,
           int faceNum,
           vector<int>& f2c );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceNum - face number ( <1 means all)
 *
 * @param f2c - vector containing the face->cell connectivity array for the
* given face (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_f2c().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Mesh::readF2N ( int simNum,
           int zoneNum,
                     int faceNum,
           vector<int>& f2n );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param faceNum - face number ( <1 means all )
 *
 * @param f2n - vector containing the face->node connectivity array for given
* face number (output)
*
* @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_f2n().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Mesh::readN2C(int simNum,
           int zoneNum,
           int nodeNum,
           vector<int>& n2c );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param nodeNum - node number ( < 1 means all )
 *
 * @param n2c - node->cell connectivity array (output)
*
* @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_read_n2c().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_MESH_H_
#define __DTF_LIB_MESH_H_

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibMesh;

class Mesh : public LibObject
{
    friend class ClassInfo_DTFLibMesh;

private:
    Mesh();
    Mesh(string className, int objectID);

    static ClassInfo_DTFLibMesh classInfo;

public:
    virtual ~Mesh();

    bool queryC2F(int simNum,
                  int zoneNum,
                  int cellNum,
                  vector<int> &facesPerCell);

    bool queryC2N(int simNum,
                  int zoneNum,
                  int cellNum,
                  vector<int> &nodesPerCell);

    bool queryC2Npos(int simNum,
                     int zoneNum,
                     int cellNum, int &offset);

    bool queryF2C(int simNum,
                  int zoneNum,
                  int faceNum,
                  int &numF2C);

    bool queryF2N(int simNum,
                  int zoneNum,
                  int faceNum,
                  vector<int> &nodesPerFace);

    bool queryF2Npos(int simNum,
                     int zoneNum,
                     int faceNum,
                     int &offset);

    bool queryN2C(int simNum,
                  int zoneNum,
                  int nodeNum,
                  vector<int> &cellsPerNode);

    bool readC2F(int simNum,
                 int zoneNum,
                 int cellNum,
                 vector<int> &c2f);

    bool readC2N(int simNum,
                 int zoneNum,
                 int cellNum,
                 vector<int> &c2n);

    bool readF2C(int simNum,
                 int zoneNum,
                 int faceNum,
                 vector<int> &f2c);

    bool readF2N(int simNum,
                 int zoneNum,
                 int faceNum,
                 vector<int> &f2n);

    bool readN2C(int simNum,
                 int zoneNum,
                 int nodeNum,
                 vector<int> &n2c);
};

CLASSINFO(ClassInfo_DTFLibMesh, Mesh);
};
#endif

/** EOC */
