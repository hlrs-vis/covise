/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class ShowL                           ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                               ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <covise/covise.h>
#include "ShowL.h"

//
// Constructor
//
ShowL::ShowL()
    : numEn_(0)
    , incAlloc_(5)
    , numAlloc_(incAlloc_)
    , val_(NULL)
{
    arr_ = new char *[incAlloc_];
    val_ = new int[incAlloc_];
}

void
ShowL::add(const char *nm, const int &val)
{

    // we may have to realloc memory
    if (numEn_ >= numAlloc_)
    {
        numAlloc_ += incAlloc_;
        char **tmpArr = new char *[numAlloc_];
        int *tmpVal = new int[numAlloc_];

        int i;
        for (i = 0; i < numEn_; ++i)
        {
            tmpVal[i] = val_[i];
            tmpArr[i] = new char[1 + strlen(arr_[i])];
            strcpy(tmpArr[i], arr_[i]);
        }
        delete[] val_;
        for (i = 0; i < numEn_; ++i)
        {
            delete[] arr_[i];
        }
        delete[] arr_;

        arr_ = tmpArr;
        val_ = tmpVal;
    }

    // we add only a new element if the key is not present
    int i;
    int notThere = 1;
    int idx = -1;
    for (i = 0; i < numEn_; ++i)
    {
        if (!strcmp(nm, arr_[i]))
        {
            notThere = 0;
            idx = i;
        }
    }
    if (notThere)
    {
        arr_[numEn_] = new char[1 + strlen(nm)];
        strcpy(arr_[numEn_], nm);
        val_[numEn_] = val;
        numEn_++;
    }
    // we set the value if chanded
    else
    {
        val_[idx] = val;
    }
}

void
ShowL::removeAll()
{
    delete[] val_;
    int i;
    for (i = 0; i < numEn_; ++i)
    {
        delete[] arr_[i];
    }
    delete[] arr_;
    numEn_ = 0;
    numAlloc_ = incAlloc_;
    arr_ = new char *[incAlloc_];
    val_ = new int[incAlloc_];
}

//
// get value
//
int
ShowL::get(const char *nm, int &ret)
{
    int there = 0;
    int idx = -1;
    int i;
    for (i = 0; i < numEn_; ++i)
    {
        char *chP = arr_[i];
        if (chP)
        {
            if (!strcmp(nm, chP))
            {
                idx = i;
                break;
            }
        }
    }

    if (idx > -1)
    {
        ret = val_[idx];
        there = 1;
    }
    else
    {
        ret = 0;
    }
    return there;
}

void
ShowL::addCoObjNm(const char *nm, const int &val)
{
    char *redNm = new char[1 + strlen(nm)];
    reduce(nm, redNm);
    add(redNm, val);
    delete[] redNm;
}

int
ShowL::getCoObjNm(const char *nm, int &ret)
{
    // int retval = 0;
    char *redNm = new char[1 + strlen(nm)];
    reduce(nm, redNm);

    if (get(redNm, ret))
    {
        return get(redNm, ret);
    }

    return get(nm, ret);
}

int
ShowL::remove(const char *nm)
{
    int there = 0;
    int idx = -1;
    int i;
    for (i = 0; i < numEn_; ++i)
    {
        if (!strcmp(nm, arr_[i]))
        {
            idx = i;
            if (idx != numEn_ - 1)
            {
                delete[] arr_[i];
                arr_[i] = new char[1 + strlen(arr_[numEn_ - 1])];
                strcpy(arr_[i], arr_[numEn_ - 1]);
            }
            delete[] arr_[numEn_ - 1];
            numEn_--;

            break;
        }
    }

    if (idx == -1)
    {
        there = 1;
    }
    return there;
}

int
ShowL::reduce(const char *nm, char *redNm)
{

    if (redNm)
    {
        char *chch = new char[1 + strlen(nm)];
        char lastCh[255][20];

        strcpy(chch, nm);
        char del[3];
        strcpy(del, "_");
        char *tok;
        tok = strtok(chch, del);
        int cnt = 0;
        while (tok)
        {
            strcpy(lastCh[cnt], tok);
            tok = strtok(NULL, del);
            cnt++;
        }
        int ii;

        strcpy(redNm, lastCh[0]);
        strcat(redNm, del);
        for (ii = 1; ii < cnt - 1; ++ii)
        {
            strcat(redNm, lastCh[ii]);
            if (ii != cnt - 2)
                strcat(redNm, del);
        }
    }
    return strlen(redNm);
}

//
// Destructor
//
ShowL::~ShowL()
{
    delete[] val_;
    int i;
    for (i = 0; i < numEn_; ++i)
    {
        delete[] arr_[i];
    }
    delete[] arr_;
}
