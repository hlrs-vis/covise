/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class En6GeoBin                       ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 15.07.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "MGeoFileBin.h"
#include "MGeoFileAsc.h"

using namespace std;

//
// Constructor
//
En6MGeoBIN::En6MGeoBIN(const coModule *mod, const string &name, EnFile::BinType binType)
    : EnFile(mod, binType)
    , indexMap_(NULL)
{

    pointObj = NULL;
    className_ = string("En6MGeoBIN");
#ifdef WIN32
    in_ = fopen(name.c_str(), "rb");
#else
    in_ = fopen(name.c_str(), "r");
#endif
    if (in_)
    {
        isOpen_ = true;
    }
    else
    {
        cerr << className_ << "::En6MGeoBIN(..) open NOT successful" << endl;
    }

    // get runtime number-representation
    // was soll das? wer braucht das?
    // wird jetzt automatisch korrigiert.

    /*int num(0);
   unsigned char t[4];

   t[0] = (unsigned char) 1;
   t[1] = '\0';
   t[2] = '\0';
   t[3] = '\0';

   //num = (int) t;
   memcpy(&num,t,sizeof(int));
   if ( num == 1 )
   {
      byteSwap_ = true;
   }
   else if ( num == 16777216 )
   {
      byteSwap_ = false;
   }
   else
   {
       cerr << className_ << "::En6MGeoBIN(..) unknown byte-order" << endl;
   }*/
}

// read header
int
En6MGeoBIN::readHeader()
{
    int ret(0);
    if (isOpen_)
    {
        string line;
        // binary type
        line = getStr();

        // 2 lines decription - ignore it
        line = getStr();
        line = getStr();

        // number of coordinates
        numCoords_ = getInt();
        cerr << className_ << "::readHeader()  got " << numCoords_ << " coordinates" << endl;
        ret = 1;
    }

    return ret;
}

// read coordinates
int
En6MGeoBIN::readCoords()
{
    int ret(0);
    // we read only if we have a valid file and coordinates
    if ((isOpen_) && (numCoords_ > 0))
    {

        // allocate arrays
        // cleanup will be made by DataCont::cleanAll()
        dc_.setNumCoord(numCoords_);
        float *tmpcoord = new float[numCoords_ * 3];
        dc_.x = new float[numCoords_];
        dc_.y = new float[numCoords_];
        dc_.z = new float[numCoords_];
        indexMap_ = new int[numCoords_];
        maxIndex_ = numCoords_;
        getIntArr(numCoords_, indexMap_);
        getFloatArr(numCoords_ * 3, tmpcoord);

        for (int i = 0; i < numCoords_; i++)
        {
            dc_.x[i] = tmpcoord[i * 3 + 0];
            dc_.y[i] = tmpcoord[i * 3 + 1];
            dc_.z[i] = tmpcoord[i * 3 + 2];
            if (dc_.x[i] > 1000000.0)
                dc_.x[i] = 0.0;
            if (dc_.y[i] > 1000000.0)
                dc_.y[i] = 0.0;
            if (dc_.z[i] > 1000000.0)
                dc_.z[i] = 0.0;

            if (dc_.x[i] < -1000000.0)
                dc_.x[i] = 0.0;
            if (dc_.y[i] < -1000000.0)
                dc_.y[i] = 0.0;
            if (dc_.z[i] < -1000000.0)
                dc_.z[i] = 0.0;
        }
        delete[] tmpcoord;
    }

    cerr << className_ << "::readCoords() got " << dc_.getNumCoord() << " coordinates" << endl;
    return ret;
}

coDistributedObject *En6MGeoBIN::getDataObject(std::string s)
{
    return new coDoPoints(s.c_str(), numCoords_, dc_.x, dc_.y, dc_.z);
}

void
En6MGeoBIN::read()
{
    readHeader();

    readCoords();
}

//
//Destructor
//
En6MGeoBIN::~En6MGeoBIN()
{
    delete[] indexMap_;
}
