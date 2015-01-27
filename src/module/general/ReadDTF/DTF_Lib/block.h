/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/block.h
 * @brief contains definition of class DTF_Lib::Block
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 * @date 5.11.2003
 * moved static member classInfo to private scope.
 */

/** @class DTF_Lib::ClassInfo_DTFLibBlock
 * @brief used to register class DTF_Lib::Block at class manager
 *
 * \b Description:
 *
 * Class defined by macro CLASSINFO(). This class is friend of
 * DTF_Lib::Block and is therefore the only class allowed to create new
 * objects. Class manager is the only class which is allowed to tell
 * ClassInfo_DTFLibBlock to create new objects of type DTF_Lib::Block.
 */

/** @class DTF_Lib::Block
 * @brief contains functions to query block informations from DTF lib.
 */

/** @fn DTF_Lib::Block::Block();
 * @brief default constructor
 *
 * \b Description:
 *
 * calls default constructor of DTF_Lib::LibObject.
 */

/** @fn DTF_Lib::Block::Block( string className, int objectID );
 * @brief constructor which initializes new objects with given object ID.
 *
 * @param objectID - unique identifier for the created object
 *
 * \b Description:
 *
 * The class manager is used to create new objects of this class. The object ID
 * is needed by the class manager to identify the object.
 */

/** @fn virtual DTF_Lib::Block::~Block();
 * @brief default destructor.
 *
 * \b Description:
 *
 * Called when objects of DTF_Lib::Block are destroyed.
 */

/** @fn bool DTF_Lib::Block::queryBlock(int simNum,
         int zoneNum,
         int blockNum,
         int& key,
         map<string, vector<int> >& minMax,
         int& numCells );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 * @param blockNum - block number
*
* @param key - key (output)
* @param minMax - min max values, stored in a stl map; keys are:
* - "i" (i-surface)
* - "j" (j-surface)
 * - "k" (k-surface)
 * (output)
 *
 * @param numCells - number of cells in the given block (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
 * this function is implemented.
 *
 * \b Description:
 *
 * wraps calls to \c dtf_query_block().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** @fn bool DTF_Lib::Block::queryNumBlocks(int simNum,
             int zoneNum,
             int& numBlocks );
 * @brief not implemented
 *
 * @param simNum - simulation number
 * @param zoneNum - zone number
 *
 * @param numBlocks - number of blocks in grid zone (output)
 *
 * @return \c false on error, \c true on success. \c false is returned until
* this function is implemented.
*
 * \b Description:
 *
 * wraps calls to \c dtf_query_nblocks().
 *
 * @attention You'll have to implement this function if you intend to use it.
 * A warning is printed for unimplemented functions.
 */

/** EOD */

/** BOC */

#ifndef __DTF_LIB_BLOCK_H
#define __DTF_LIB_BLOCK_H

#include "baseinc.h"

namespace DTF_Lib
{
class ClassInfo_DTFLibBlock;

class Block : public LibObject
{
    friend class ClassInfo_DTFLibBlock;

protected:
    Block();
    Block(string className, int objectID);

    static ClassInfo_DTFLibBlock classInfo;

public:
    virtual ~Block();

    bool queryBlock(int simNum,
                    int zoneNum,
                    int blockNum,
                    int &key,
                    map<string, vector<int> > &minMax,
                    int &numCells);

    bool queryNumBlocks(int simNum,
                        int zoneNum,
                        int &numBlocks);
};

CLASSINFO(ClassInfo_DTFLibBlock, Block);
};
#endif

/** EOC */
