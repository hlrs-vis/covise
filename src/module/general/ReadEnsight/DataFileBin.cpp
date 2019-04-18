/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class DataFileBin                     ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 07.06.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "DataFileBin.h"
#include "EnElement.h"

#include <util/coviseCompat.h>
#include <do/coDoData.h>

//
// Constructor
//
DataFileBin::DataFileBin(ReadEnsight *mod)
    : EnFile(mod)
    , dim_(1)
    , lineCnt_(0)
    , numVals_(0)
    , indexMap_(NULL)
{
    className_ = string("DataFileBin");
}

DataFileBin::DataFileBin(ReadEnsight *mod, const string &name,
                         const int &dim,
                         const int &numVals,
                         const EnFile::BinType &binType)
    : EnFile(mod, name, dim, binType)
    , lineCnt_(0)
    , numVals_(numVals)
    , indexMap_(NULL)
{
    dim_ = dim;
    className_ = string("DataFileBin");
    byteSwap_ = false;

    // if numVals == 0 the memory in tha data container will be allocated
    // dynamically
    if (numVals_ > 0)
    {
        switch (dim_)
        {
        case 1:
            dc_.setNumCoord(numVals_);
            dc_.x = new float[numVals_];
            break;
        case 3:
            dc_.setNumCoord(numVals_);
            dc_.x = new float[numVals_];
            dc_.y = new float[numVals_];
            dc_.z = new float[numVals_];
            break;
        }
    }
}

//
// read data
//
void
DataFileBin::read(dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep, int numTimeSteps)
{
    if (isOpen_)
    {

        // Descr line
        getStr();

        // data dim
        int dataDim(dim_ * numVals_);

        float *locArr = new float[dataDim];
        getFloatArr(dataDim, locArr);

        switch (dim_)
        {
        case 1:
            memcpy(dc_.x, locArr, dataDim * sizeof(float));
            break;
        case 3:
            int cnt(0);
            for (int i = 0; i < dataDim; i += 3)
            {
                dc_.x[cnt] = locArr[i];
                dc_.y[cnt] = locArr[i + 1];
                dc_.z[cnt] = locArr[i + 2];
                cnt++;
            }
            break;
        }
        delete[] locArr;
    }
    createDataOutObj(dim, outObjects, baseName, timeStep,numTimeSteps);
}

coDistributedObject *DataFileBin::getDataObject(std::string s)
{
    if (dim_ == 1)
        return new coDoFloat(s.c_str(), numVals_, dc_.x);
    else
        return new coDoVec3(s.c_str(), numVals_, dc_.x, dc_.y, dc_.z);
}

//
// reads cell based data
//
void
DataFileBin::readCells(dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep, int numTimeSteps)
{
    EnPart *actPart;
    int numGot2d = 0, numGot3d = 0;

    if (isOpen_)
    {
        // 1 lines decription - ignore it
        getStr();
        // part key
        while (true)
        {
            string str;
            try
            {
                str = getStr();
            }
            catch (InvalidWordException e)
            {
            }
            // quit loop
            if (feof(in_))
                break;

            string partStr(str.substr(0, 4));
            if (partStr != string("part"))
                cerr << "DataFileBin::readCells(): part string not found <" << partStr << ">" << endl;

            size_t anf(str.find_first_not_of(" ", 4));
            string s(str.substr(anf, 8));
            int actPartNr(atoi(s.c_str()));
            actPart = findPart(actPartNr);
            numGot2d = 0;
            numGot3d = 0;

            bool actPartActive = false;
            if (actPart != NULL)
            {
                actPartActive = actPart->isActive();
                if (actPartActive)
                {
                    // allocate memory for a whole part
                    switch (dim_)
                    {
                    case 1:
                        // create mem for 2d/3d data
                        actPart->d2dx_ = new float[actPart->numEleRead2d()];
                        actPart->d2dy_ = NULL;
                        actPart->d2dz_ = NULL;
                        actPart->d3dx_ = new float[actPart->numEleRead3d()];
                        actPart->d3dy_ = NULL;
                        actPart->d3dz_ = NULL;
                        break;
                    case 3:
                        actPart->d2dx_ = new float[actPart->numEleRead2d()];
                        actPart->d2dy_ = new float[actPart->numEleRead2d()];
                        actPart->d2dz_ = new float[actPart->numEleRead2d()];
                        actPart->d3dx_ = new float[actPart->numEleRead3d()];
                        actPart->d3dy_ = new float[actPart->numEleRead3d()];
                        actPart->d3dz_ = new float[actPart->numEleRead3d()];
                        break;
                    }
                    resetPart(actPartNr, actPart);
                }
                // fill arrays for each element
                int cnt(0);
                int ii = 0;
                float *tempCoords = NULL;
                int startMark2d = 0, startMark3d = 0;
                for (ii = 0; ii < actPart->getNumEle(); ++ii)
                {
                    string elementType(strip(getStr().substr(0, 8)));
                    EnElement actEle(actPart->findElement(elementType));
                    if (actEle.valid())
                    {
                        int nCellsPerEle(actPart->getElementNum(elementType));
                        switch (dim_)
                        {
                        case 1:
                            if (actPartActive)
                            {
                                tempCoords = new float[nCellsPerEle];
                                getFloatArr(nCellsPerEle, tempCoords);
                                if (actEle.getDim() == EnElement::D2)
                                {
                                    std::copy(tempCoords, tempCoords + nCellsPerEle, actPart->d2dx_ + startMark2d);
                                    startMark2d += nCellsPerEle;
                                }
                                if (actEle.getDim() == EnElement::D3)
                                {
                                    std::copy(tempCoords, tempCoords + nCellsPerEle, actPart->d3dx_ + startMark3d);
                                    startMark3d += nCellsPerEle;
                                }
                                delete[] tempCoords;
                            }
                            else
                            {
                                skipFloat(nCellsPerEle);
                            }
                            break;
                        case 3:
                            if (actPartActive)
                            {
                                // here we have to disperse the arrays
                                tempCoords = new float[3 * nCellsPerEle];
                                getFloatArr(3 * nCellsPerEle, tempCoords);
                                int i, j(0);
                                if (actEle.getDim() == EnElement::D2)
                                {
                                    for (i = 0; i < nCellsPerEle; ++i)
                                    {
                                        actPart->d2dx_[cnt + i] = tempCoords[j];
                                        j++;
                                        actPart->d2dy_[cnt + i] = tempCoords[j];
                                        j++;
                                        actPart->d2dz_[cnt + i] = tempCoords[j];
                                        j++;
                                    }
                                }
                                if (actEle.getDim() == EnElement::D3)
                                {
                                    for (i = 0; i < nCellsPerEle; ++i)
                                    {
                                        actPart->d3dx_[cnt + i] = tempCoords[j];
                                        j++;
                                        actPart->d3dy_[cnt + i] = tempCoords[j];
                                        j++;
                                        actPart->d3dz_[cnt + i] = tempCoords[j];
                                        j++;
                                    }
                                }
                                if (tempCoords != NULL)
                                    delete[] tempCoords;
                            }
                            else
                                skipFloat(3 * nCellsPerEle);
                            break;
                        }
                        cnt += nCellsPerEle;
                    }
                    else
                        cerr << "DataFileBin::readCells(): actEle not valid" << endl;
                }
            }
            else
                cerr << "DataFileBin::readCells(): part not found part nr: " << actPartNr << endl;
        }
    }
    createDataOutObj(dim, outObjects, baseName, timeStep,false);
}
