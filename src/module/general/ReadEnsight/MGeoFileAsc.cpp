/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class       EnGeoFile                 ++
// ++                                           En6MGeoASC                 ++
// ++                                                                     ++
// ++ Author of initial version:  Ralf Mikulla (rm@vircinity.com)         ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 05.06.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include "MGeoFileAsc.h"
#include "EnElement.h"

#include <util/coviseCompat.h>
#include <api/coModule.h>

using namespace std;

const int lineLen(250);
//////////////////////// geo file class ////////////////////////////////////

//
// Constructor
//
En6MGeoASC::En6MGeoASC(const coModule *mod)
    : EnFile(mod)
    , lineCnt_(0)
    , numCoords_(0)
    , indexMap_(NULL)
    , maxIndex_(0)
    , lastNc_(0)
{
    className_ = string("En6MGeoASC");
}

En6MGeoASC::En6MGeoASC(const coModule *mod, const string &name)
    : EnFile(mod, name)
    , lineCnt_(0)
    , numCoords_(0)
    , indexMap_(NULL)
    , maxIndex_(0)
    , lastNc_(0)
{
    className_ = string("En6MGeoASC");
    pointObj = NULL;
}

//
// Destructor
//
En6MGeoASC::~En6MGeoASC()
{
    delete[] indexMap_;
}

void
En6MGeoASC::read()
{
    // read header
    readHeader();
    // read coordinates
    En6MGeoASC::readCoords();
    // TBD: err handling
}

// read header
int
En6MGeoASC::readHeader()
{
    int ret(0);
    if (isOpen_)
    {
        char buf[lineLen];
        // 2 lines decription - ignore it
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        // node id
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string tmp(buf);
        char tok[20];
        strcpy(tok, " id");
        size_t beg(tmp.find(tok));

        if (beg == string::npos)
        {
            cerr << className_ << "::readHeader() ERROR node-id not found" << endl;
            return ret;
        }
        // " id"  has length 3
        beg += 4;
        string cmpStr(strip(tmp.substr(beg)));
        nodeId_ = -1; ////////
        if (cmpStr == string("off"))
        {
            nodeId_ = EnFile::OFF;
        }
        else if (cmpStr == string("given"))
        {
            nodeId_ = EnFile::GIVEN;
        }
        else if (cmpStr == string("assign"))
        {
            nodeId_ = EnFile::ASSIGN;
        }
        else if (cmpStr == string("ignore"))
        {
            nodeId_ = EnFile::EN_IGNORE;
        }

        // element id
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        tmp = string(buf);

        beg = tmp.find(tok);

        if (beg == string::npos)
        {
            cerr << className_ << "::readHeader() ERROR element-id not found" << endl;
            return ret;
        }

        beg += 4;
        cmpStr = strip(tmp.substr(beg));
        if (cmpStr == string("off"))
        {
            elementId_ = EnFile::OFF;
        }
        else if (cmpStr == string("given"))
        {
            elementId_ = EnFile::GIVEN;
        }
        else if (cmpStr == string("assign"))
        {
            elementId_ = EnFile::ASSIGN;
        }
        else if (cmpStr == string("ignore"))
        {
            elementId_ = EnFile::EN_IGNORE;
        }
        else
        {
            cerr << className_ << "::readHeader()  ELEMENT-ID Error" << endl;
        }

        ret = 1;
    }

    return ret;
}

// read coordinates
int
En6MGeoASC::readCoords()
{
    int ret(0);

    if (isOpen_)
    {
        char buf[lineLen];
        // coordinates token
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string tmp(buf);
        char tok[20];
        strcpy(tok, "coordinates");
        size_t beg(tmp.find_first_of(tok));

        if (beg == string::npos)
        {
            cerr << className_ << "::readCoords() ERROR coordinates token not found" << endl;
            return ret;
        }

        // number of coordinates
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        tmp = string(buf);

        string num(strip(tmp));
        if (num.empty())
        {
            cerr << className_ << "::readCoords() ERROR could not read number of coordinates" << endl;
            return ret;
        }

        numCoords_ = atoi(num.c_str());
        //cerr << className_ << "::readCoords()  NUM COORDINATES " << numCoords_ << endl;
    }

    // we read only if we have a valid file and coordinates
    if ((isOpen_) && (numCoords_ > 0))
    {

        char buf[lineLen];

        // allocate arrays
        dc_.setNumCoord(numCoords_);
        dc_.x = new float[numCoords_];
        dc_.y = new float[numCoords_];
        dc_.z = new float[numCoords_];
        indexMap_ = new int[numCoords_];
        maxIndex_ = numCoords_ - 1;

        // read all coordinates
        int i;
        for (i = 0; i < numCoords_; ++i)
        {
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            //	    string tmp(buf);
            float x, y, z;
            int idx, entries;
            switch (nodeId_)
            {
            case ASSIGN:
                entries = sscanf(buf, "%e%e%e", &x, &y, &z);
                if (entries != 3)
                {
                    cerr << className_ << "::readCoords()  ERROR reading coordinates " << endl;
                    cerr << className_ << "::readCoords()  line: " << lineCnt_ << " expect 3 entries have " << entries << endl;
                    // TBD: insert error handler
                }
                // fill container
                dc_.x[i] = x;
                dc_.y[i] = y;
                dc_.z[i] = z;
                break;
            case GIVEN:
                entries = sscanf(buf, "%d%e%e%e", &idx, &x, &y, &z);
                if (entries != 4)
                {
                    cerr << className_ << "::readCoords()  ERROR reading coordinates " << endl;
                    cerr << className_ << "::readCoords()  line: " << lineCnt_ << " expect 3 entries have " << entries << endl;
                }
                fillIndexMap(idx, i);
                // fill container
                dc_.x[i] = x;
                dc_.y[i] = y;
                dc_.z[i] = z;
                break;
            default:
                return -1;
                // error
                break;
            }
        }
    }
    return ret;
}

coDistributedObject *En6MGeoASC::getDataObject(std::string s)
{
    return new coDoPoints(s.c_str(), numCoords_, dc_.x, dc_.y, dc_.z);
}

// helper converts char buf containing num ints of length int_leng to int-array arr
void
En6MGeoASC::atoiArr(const int &int_leng, char *buf, int *arr, const int &num)
{
    if ((buf != NULL) && (arr != NULL))
    {
        int cnt = 0, i = 0;
        string str(buf);
        //cerr << "En6MGeoASC::atoiArr(..) STR: " << str << endl;
        string::iterator it = str.begin();
        string chunk;
        while (it != str.end())
        {
            char x = *it;
            if (x != ' ')
                chunk += x;
            if (((x == ' ') && (!chunk.empty())) || (cnt >= int_leng))
            {
                arr[i] = atoi(chunk.c_str());
                //cerr << "chunk: " << chunk << " int: " << i << " val: " << arr[i] << endl;
                cnt = 0;
                // chunk.clear() does not wor on linux
                chunk.erase(chunk.begin(), chunk.end());
                i++;
            }
            ++it;
            ++cnt;
        }
        // this warning should never occur
        if (i > num)
            cerr << "En6MGeoASC::atoiArr(..) WARNING: found more numberes than expected!" << endl;
        // chunk will not be empty
        if (!chunk.empty())
            arr[i] = atoi(chunk.c_str());
        //cerr << "chunk: " << chunk << " int: " << i << " val: " << arr[i] << endl;
    }
}

void
En6MGeoASC::fillIndexMap(const int &i, const int &natIdx)
{
    //const int offSet(100);
    // initial
    if (maxIndex_ == 0)
    {
        maxIndex_ = numCoords_;
        indexMap_ = new int[maxIndex_];
    }
    // realloc
    if (i >= maxIndex_)
    {
        //int * tmp = new int[i+offSet];
        int newSize = (int)(i * 1.25);
        int *tmp = new int[newSize];
        int j;
        for (j = 0; j < maxIndex_; ++j)
            tmp[j] = indexMap_[j];
        maxIndex_ = newSize;
        delete[] indexMap_;
        indexMap_ = tmp;
    }
    indexMap_[i] = natIdx;
}

#ifdef TESTING

main(int argc, char **argv)
{

    if (argc != 2)
    {
        cerr << " Test (En6MGeoASC) number of arguments wrong: expect only filename" << endl;
        exit(-1);
    }

    string filen(argv[1]);

    cerr << " Test (En6MGeoASC) try to open file <" << filen << "> " << endl;
    En6MGeoASC enf(filen);

    enf.read();
}
#endif
