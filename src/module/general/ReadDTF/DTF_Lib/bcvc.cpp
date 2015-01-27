/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** BOD */

/** @file DTF_Lib/bcvc.cpp
 * @brief contains definition of methods for class DTF_Lib::BcVc.
 * @author Alexander Martinez <kubus3561@gmx.de>
 * @date 01.10.2003
 * created
 */

/** EOD */

/** BOC */

#include "bcvc.h"

using namespace DTF_Lib;

CLASSINFO_OBJ(ClassInfo_DTFLibBcVc, BcVc, "DTF_Lib::BcVc", 1);

BcVc::BcVc()
    : LibObject(){};

BcVc::BcVc(string className, int objectID)
    : LibObject(className, objectID)
{
    INC_OBJ_COUNT(getClassName());
};

BcVc::~BcVc()
{
    clear();
    DEC_OBJ_COUNT(getClassName());
}

bool BcVc::queryBFnum(int simNum,
                      int zoneNum,
                      int faceIndex,
                      int &bfNum)
{
    return implementMe();
}

bool BcVc::queryIFnum(int simNum,
                      int zoneNum,
                      int faceIndex,
                      int &ifNum)
{
    return implementMe();
}

bool BcVc::queryBF2BCR(int simNum,
                       int zoneNum,
                       vector<int> &bfacesOfType)
{
    return implementMe();
}

bool BcVc::queryXF2BCR(int simNum,
                       int zoneNum,
                       vector<int> &xfacesOfType)
{
    return implementMe();
}

bool BcVc::readBF2BCR(int simNum,
                      int zoneNum,
                      int faceNum,
                      vector<int> &bf2f,
                      vector<int> &bf2r)
{
    return implementMe();
}

bool BcVc::readBF2NBCR(int simNum,
                       int zoneNum,
                       int faceNum,
                       vector<int> &bf2n,
                       vector<int> &bf2r)
{
    return implementMe();
}

bool BcVc::readXF2BCR(int simNum,
                      int zoneNum,
                      int faceNum,
                      vector<int> &xf2f,
                      vector<int> &xf2r)
{
    return implementMe();
}

bool BcVc::readXF2NBCR(int simNum,
                       int zoneNum,
                       int faceNum,
                       vector<int> &xf2n,
                       vector<int> &xf2r)
{
    return implementMe();
}

/** EOC */
