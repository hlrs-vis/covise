/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                           (C)2002 / 2003 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class Reducer                         ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 11.06.2003                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <util/coviseCompat.h>
#ifdef WIN32
typedef unsigned int uint;
#endif
#include "Reducer.h"

// prevent from use
Reducer::Reducer(const Reducer &r)
    : dc_(r.dc_)
    , idxMap_(NULL)
{
}

const Reducer &
    Reducer::
    operator=(const Reducer &r) { return r; }

//
// Constructor
//
Reducer::Reducer(DataCont &dc, int *im)
    : dc_(dc)
    , idxMap_(im)
{
}

//
// Method
//
int
Reducer::removeUnused(float **xn, float **yn, float **zn)
{
    // check if we got a complete container
    if ((dc_.x == NULL) || (dc_.y == NULL) || (dc_.z == NULL) || (dc_.el == NULL)
        || (dc_.el == NULL) || (dc_.cl == NULL) || (dc_.tl == NULL))
    {
        cerr << "Reducer::removeUnused( ) got NULL array pointer " << endl;
        return 0;
    }

    // 0= unused 1=used
    int numCoord = dc_.getNumCoord();
    unsigned int *blacklist = NULL;
    try
    {
        blacklist = new unsigned int[numCoord];
    }
    catch (std::exception &e)
    {
        cout << "Exception (alloc of locArr Line: __LINE__) : " << e.what();
    }

    int i;

    try
    {
        if (idxMap_ == NULL)
            idxMap_ = new int[numCoord];
    }
    catch (std::exception &e)
    {
        cout << "Exception (alloc of locArr Line: __LINE__) : " << e.what();
    }

    for (i = 0; i < numCoord; ++i)
    {
        blacklist[i] = 0;
        idxMap_[i] = -1;
    }

    int maxCorn(dc_.getNumConn());
    int redCnt(0);

    for (i = 0; i < maxCorn; ++i)
        blacklist[dc_.cl[i]] = 1;

    for (i = 0; i < numCoord; ++i)
    {
        if (blacklist[i] == 1)
            redCnt++;
    }

    // shortend coordinate arrays
    float *x = NULL, *y = NULL, *z = NULL;
    try
    {
        x = new float[redCnt];
        y = new float[redCnt];
        z = new float[redCnt];
    }
    catch (std::exception &e)
    {
        cout << "Exception (alloc of locArr Line: __LINE__) : " << e.what();
    }

    int cnt(0);

    for (i = 0; i < numCoord; ++i)
    {
        if (blacklist[i] == 1)
        {
            x[cnt] = dc_.x[i];
            y[cnt] = dc_.y[i];
            z[cnt] = dc_.z[i];
            idxMap_[i] = cnt;
            cnt++;
        }
    }

    // use new coordinate indindicees in corner list
    int tmp;
    for (i = 0; i < maxCorn; ++i)
    {
        tmp = idxMap_[dc_.cl[i]];
        dc_.cl[i] = tmp;
    }

    delete[] blacklist;
    // rebuild DataCont
    bool mkNew = (xn != NULL) && (yn != NULL) && (zn != NULL);
    if (mkNew)
    {
        *xn = x;
        *yn = y;
        *zn = z;
    }
    else
    {
        delete[] dc_.x;
        delete[] dc_.y;
        delete[] dc_.z;

        dc_.x = x;
        dc_.y = y;
        dc_.z = z;
    }
    dc_.setNumCoord(redCnt);

    return numCoord - redCnt - 1;
}

DataCont
Reducer::reduceAndCopyData()
{
    DataCont ret;
    if (idxMap_ != NULL)
    {
        int i;
        int numRed(0);
        int numCoord(dc_.getNumCoord());

        for (i = 0; i < numCoord; ++i)
        {
            if (idxMap_[i] >= 0)
                numRed++;
        }
        float *x = NULL, *y = NULL, *z = NULL;
        if (dc_.x != NULL)
            x = new float[numRed];
        if (dc_.y != NULL)
            y = new float[numRed];
        if (dc_.z != NULL)
            z = new float[numRed];

        int cnt(0);
        int idx;
        for (i = 0; i < numCoord; ++i)
        {
            idx = idxMap_[i];
            if (idx >= 0)
            {
                if (dc_.x != NULL)
                    x[cnt] = dc_.x[i];
                if (dc_.y != NULL)
                    y[cnt] = dc_.y[i];
                if (dc_.z != NULL)
                    z[cnt] = dc_.z[i];
                cnt++;
            }
        }
        ret.setNumCoord(numRed);
        if (dc_.x != NULL)
        {
            ret.x = x;
        }
        if (dc_.y != NULL)
        {
            ret.y = y;
        }
        if (dc_.z != NULL)
        {
            ret.z = z;
        }

        return ret;
    }
    else
    {
        int numCoord(dc_.getNumCoord());
        if (dc_.x != NULL)
            ret.x = new float[numCoord];
        if (dc_.y != NULL)
            ret.y = new float[numCoord];
        if (dc_.z != NULL)
            ret.z = new float[numCoord];
        for (int i = 0; i < numCoord; ++i)
        {
            if (dc_.x != NULL)
                ret.x[i] = dc_.x[i];
            if (dc_.y != NULL)
                ret.y[i] = dc_.y[i];
            if (dc_.z != NULL)
                ret.z[i] = dc_.z[i];
        }
        ret.setNumCoord(numCoord);
    }
    return ret;
}

int
Reducer::reduceData()
{
    if (idxMap_ != NULL)
    {
        int i;
        int numRed(0);
        int numCoord(dc_.getNumCoord());

        for (i = 0; i < numCoord; ++i)
        {
            if (idxMap_[i] >= 0)
                numRed++;
        }
        float *x = NULL, *y = NULL, *z = NULL;
        if (dc_.x != NULL)
            x = new float[numRed];
        if (dc_.y != NULL)
            y = new float[numRed];
        if (dc_.z != NULL)
            z = new float[numRed];

        int cnt(0);
        int idx;
        for (i = 0; i < numCoord; ++i)
        {
            idx = idxMap_[i];
            if (idx >= 0)
            {
                if (dc_.x != NULL)
                    x[cnt] = dc_.x[i];
                if (dc_.y != NULL)
                    y[cnt] = dc_.y[i];
                if (dc_.z != NULL)
                    z[cnt] = dc_.z[i];
                cnt++;
            }
        }
        dc_.setNumCoord(numRed);

        if (dc_.x != NULL)
        {
            delete[] dc_.x;
            dc_.x = x;
        }
        if (dc_.y != NULL)
        {
            delete[] dc_.y;
            dc_.y = y;
        }
        if (dc_.z != NULL)
        {
            delete[] dc_.z;
            dc_.z = z;
        }
    }
    return 0;
}

const int *
Reducer::getIdxMap()
{
    return idxMap_;
}

//
// Destructor
//
Reducer::~Reducer()
{
}
