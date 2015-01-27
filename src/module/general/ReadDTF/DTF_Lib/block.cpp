/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/block.cpp
 * @brief contains definition of methods for class DTF_Lib::Block.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "block.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibBlock, Block, "DTF_Lib::Block", 1);

Block::Block()
    : LibObject(){};

Block::Block(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
}

Block::~Block()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool Block::queryBlock(int simNum,
                       int zoneNum,
                       int blockNum,
                       int &key,
                       map<string, vector<int> > &minMax,
                       int &numCells)
{
    return implementMe();
}

bool Block::queryNumBlocks(int simNum,
                           int zoneNum,
                           int &numBlocks)
{
    return implementMe();
}

/** EOC */
