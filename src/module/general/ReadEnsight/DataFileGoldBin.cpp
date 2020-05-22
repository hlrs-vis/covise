/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2003 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class DataFileGoldBin                    ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 07.04.2003                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "DataFileGoldBin.h"
#include "ReadEnsight.h"
#include <util/byteswap.h>

//
// Constructor
//
DataFileGoldBin::DataFileGoldBin(ReadEnsight *mod,
                                 const string &name,
                                 const int &dim,
                                 const int &numVals,
                                 const EnFile::BinType &binType)
    : EnFile(mod, name, dim, binType)
    , lineCnt_(0)
    , numVals_(numVals)
    , indexMap_(NULL)
    , actPartIndex_(0)
{
    className_ = string("DataFileGoldBin");

    byteSwap_ = false;
    //byteSwap_ =  machineIsLittleEndian();

    // dc_ is only a marker here
    switch (dim_)
    {
    case 1:
        dc_.setNumCoord(numVals_);
        dc_.x = new float[1];
        break;
    case 3:
        dc_.setNumCoord(numVals_);
        dc_.x = new float[1];
        dc_.y = new float[1];
        dc_.z = new float[1];
        break;
    }
}

void
DataFileGoldBin::readCells(dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep, int numTimeSteps)
{
    if (isOpen_)
    {
        bool written = false;
        size_t id(0);
        int actPartNr;
        EnPart *actPart(NULL);
        int eleCnt2d = 0, eleCnt3d = 0;

        // 1 lines decription - ignore it
        string currentLine(getStr());

        while (!feof(in_)) // read all timesteps
        {
            size_t tt = currentLine.find("END TIME STEP");
            if (tt != string::npos)
            {
                currentLine = getStr(); // read description
                                        // create DO's

                coDistributedObject **oOut = ens->createDataOutObj(dim, baseName, dc_, timeStep,false);


                if (oOut[0] != NULL)
                    outObjects[timeStep] = oOut[0];

                timeStep++;
                if(timeStep < ens->globalParts_.size())
                {
                    setPartList(&ens->globalParts_[timeStep]); // make sure we use the current part list for this timestep
                }
                written = true;
            }
            tt = currentLine.find("BEGIN TIME STEP");
            if (tt != string::npos)
            {
                currentLine = getStr(); // read description
            }

            currentLine = getStr(); // this should be the part line or an element name

            id = currentLine.find("part");
            if (id != string::npos)
            {
                // part line found   -- we skip it for the moment --
                // length of "part" +1
                actPartNr = getInt();

                if (actPartNr > 10000 || actPartNr < 0)
                {
                    byteSwap_ = !byteSwap_;
                    byteSwap(actPartNr);
                }
                actPart = NULL;
                actPart = findPart(actPartNr);
                // allocate memory for whole parts
                if (actPart != NULL)
                {
                    if (actPart->isActive())
                    {
                        eleCnt2d = 0;
                        eleCnt3d = 0;
                        switch (dim_)
                        {
                        case 1:
                            // create mem for 2d/3d data
                            // we allocate one element more in order to have a valid field
                            // even if actPart->numEleRead?? is zero
                            actPart->d2dx_ = new float[actPart->numEleRead2d() + 1];
                            actPart->d2dy_ = NULL;
                            actPart->d2dz_ = NULL;
                            actPart->d3dx_ = new float[actPart->numEleRead3d() + 1];
                            actPart->d3dy_ = NULL;
                            actPart->d3dz_ = NULL;
                            break;
                        case 3:
                            actPart->d2dx_ = new float[actPart->numEleRead2d() + 1];
                            actPart->d2dy_ = new float[actPart->numEleRead2d() + 1];
                            actPart->d2dz_ = new float[actPart->numEleRead2d() + 1];
                            actPart->d3dx_ = new float[actPart->numEleRead3d() + 1];
                            actPart->d3dy_ = new float[actPart->numEleRead3d() + 1];
                            actPart->d3dz_ = new float[actPart->numEleRead3d() + 1];
                            break;
                        }
                        // 			cerr << "DataFileGoldBin::readCells() actPart->d2dx_ " << actPart->d2dx_;
                        // 			cerr << " PART " << actPartNr << endl;
                        // 			cerr << "actPart->d3dx_ " << actPart->d3dx_;
                        // 			cerr << " PART " << actPartNr << endl;
                    }
                    else
                    {
                        actPart->d2dx_ = NULL;
                        actPart->d2dy_ = NULL;
                        actPart->d2dz_ = NULL;
                        actPart->d3dx_ = NULL;
                        actPart->d3dy_ = NULL;
                        actPart->d3dz_ = NULL;
                    }
                }
                currentLine = getStr(); // this should be the elementName now
            }

            string elemName(currentLine); 

            if ((actPart != NULL) && (actPart->isActive()))
            {
                string elementType(strip(elemName));
                EnElement elem(elementType);
                int anzEle(0);
                // we have a valid ENSIGHT element
                if (elem.valid())
                {
                    anzEle = actPart->getElementNum(elementType);
                    //cerr << "DataFileGoldBin::readCells() " << anzEle << " for " << elementType  << endl;
                    EnElement thisEle = actPart->findElement(elementType);
                    vector<int> bl(thisEle.getBlacklist());
                    if (thisEle.getBlacklist().size() != anzEle)
                    {
                        //cerr << "DataFileGoldBin::readCells( ) blacklist size problem " << bl.size() << endl;
                    }

                    int i;
                    float *tArr1 = NULL, *tArr2 = NULL, *tArr3 = NULL;
                    switch (dim_)
                    {
                    case 1:
                        tArr1 = new float[anzEle];
                        // scalar data
                        getFloatArr(anzEle, tArr1);
                        if (thisEle.getDim() == EnElement::D2)
                        {
                            for (i = 0; i < anzEle; ++i)
                            {
                                if (bl[i] > 0)
                                {
                                    actPart->d2dx_[eleCnt2d] = tArr1[i];
                                    ++eleCnt2d;
                                }
                            }
                        }
                        else if (thisEle.getDim() == EnElement::D3)
                        {
                            for (i = 0; i < anzEle; ++i)
                            {
                                if (bl[i] > 0)
                                {
                                    actPart->d3dx_[eleCnt3d] = tArr1[i];
                                    ++eleCnt3d;
                                }
                            }
                        }
                        // 			cerr << " EleCnt2d: " << eleCnt2d;
                        // 			cerr << " ACT-PART numEleRead2d " <<  actPart->numEleRead2d();
                        // 			cerr << endl;

                        // 			cerr << " EleCnt3d: " << eleCnt3d;
                        // 			cerr << " ACT-PART numEleRead3d " <<  actPart->numEleRead3d();
                        // 			cerr << endl;
                        delete[] tArr1;
                        break;
                    case 3:
                        tArr1 = new float[anzEle];
                        tArr2 = new float[anzEle];
                        tArr3 = new float[anzEle];
                        // 3-dim vector data
                        getFloatArr(anzEle, tArr1);
                        getFloatArr(anzEle, tArr2);
                        getFloatArr(anzEle, tArr3);
                        if (thisEle.getDim() == EnElement::D2)
                        {
                            for (i = 0; i < anzEle; ++i)
                            {
                                if (bl[i] > 0)
                                {
                                    actPart->d2dx_[eleCnt2d] = tArr1[i];
                                    actPart->d2dy_[eleCnt2d] = tArr2[i];
                                    actPart->d2dz_[eleCnt2d] = tArr3[i];
                                    ++eleCnt2d;
                                }
                            }
                        }
                        else if (thisEle.getDim() == EnElement::D3)
                        {
                            for (i = 0; i < anzEle; ++i)
                            {
                                if (bl[i] > 0)
                                {
                                    actPart->d3dx_[eleCnt3d] = tArr1[i];
                                    actPart->d3dy_[eleCnt3d] = tArr2[i];
                                    actPart->d3dz_[eleCnt3d] = tArr3[i];
                                    ++eleCnt3d;
                                }
                            }
                        }
                        delete[] tArr1;
                        delete[] tArr2;
                        delete[] tArr3;
                        break;
                    }
                } // if( elem.valid()

            } // if ( actPart->isActive()
            // the part was not read during the geometry sweep
            else
            {
                EnPart currPart;
                if (actPart == NULL)
                {
                    // Actually, actPart should never be NULL since all (even deactivated) parts were added to the partList.
                    currPart = findMasterPart(actPartNr);
                    if (currPart.getPartNum() == 0)
                        cerr << "DataFileGoldBin::readCells() part with number " << actPartNr << "was not found in the master-part-list" << endl << "serious ERROR" << endl;
                }
                else
                {
                    currPart = *actPart;
                }

                int numParts = currPart.getNumEle();
                // skip data
                while ((!feof(in_)) && (numParts > 0))
                {
                    currentLine = getStr();
                    string elementType(strip(currentLine));
                    EnElement elem(elementType);
                    int anzEle(0);
                    // we have a valid ENSIGHT element
                    if (elem.valid())
                    {
                        anzEle = currPart.getElementNum(elementType);
                        switch (dim_)
                        {
                        case 1:
                            // scalar data
                            skipFloat(anzEle);
                            break;
                        case 3:
                            // 3-dim vector data
                            skipFloat(anzEle);
                            skipFloat(anzEle);
                            skipFloat(anzEle);
                            break;
                        }
                    }
                    numParts--;
                }
            }

        }
        if (written)
        {
            //dc_.cleanAll();
        }
        else
            createDataOutObj(dim, outObjects, baseName, timeStep,numTimeSteps,false);
    }

}

//
// Method
//
void
DataFileGoldBin::read(dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep, int numTimeSteps)
{
    if (isOpen_)
    {
        size_t id(0);
        int actPartNr;

        while (!feof(in_))
        {
            float *arr1 = NULL, *arr2 = NULL, *arr3 = NULL;
            int numVal;
            string tmp(getStr());
            // we only read the first time step
            if (tmp.find("END TIME STEP") != string::npos)
                break;
            // any text including the word "part" will disturb the passing
            // work around for particles
            if (tmp.find("particles") != string::npos)
            {
                tmp = getStr();
            }
            EnPart *actPart = NULL;
            // we found a part line
            id = tmp.find("part");
            if (id != string::npos)
            {
                // part line found   -- we skip it for the moment --
                actPartNr = getInt();
                if (actPartNr > 10000 || actPartNr < 0)
                {
                    byteSwap_ = !byteSwap_;
                    byteSwap(actPartNr);
                }
                // cerr << "DataFileGoldBin::read( ) part " <<  actPartNr << endl;
                actPart = findPart(actPartNr);

                // allocate memory for whole parts
                if (actPart != NULL)
                {
                    numVal = actPart->numCoords();
                    if (actPart->isActive())
                    {
                        switch (dim_)
                        {
                        case 1:
                            //cerr << "DataFileGoldBin::read() allocate 1-dim arr" << endl;
                            arr1 = new float[numVal];
                            // if pointer is not copied sometimes: free data here
                            if (actPart->arr1_ != NULL)
                                delete[] actPart -> arr1_;
                            actPart->arr1_ = arr1;
                            break;
                        case 3:
                            //cerr << "DataFileGoldBin::read() allocate 3-dim arr" << endl;
                            arr1 = new float[numVal];
                            arr2 = new float[numVal];
                            arr3 = new float[numVal];

                            actPart->arr1_ = arr1;
                            actPart->arr2_ = arr2;
                            actPart->arr3_ = arr3;
                            break;
                        }
                    }
                }
                else // Actually, actPart should never be NULL since all (even deactivated) parts were added to the partList.
                {
                    EnPart altPart = findMasterPart(actPartNr);
                    if (altPart.getPartNum() > 0)
                        numVal = altPart.numCoords();
                    else
                        cerr << "DataFileGoldBin::read() part with number " << actPartNr << "was not found in the master-part-list" << endl << "serious ERROR" << endl;
                }

                tmp = getStr();
                id = tmp.find("coordinates");
                if (id != string::npos)
                {
                    if ((actPart != NULL) && (actPart->isActive()))
                    {
                        // coordinates -line
                        switch (dim_)
                        {
                        case 1:
                            //cerr << "DataFileGoldBin::read()  reading scalar data" << endl;
                            getFloatArr(numVal, arr1);
                            break;
                        case 3:
                            getFloatArr(numVal, arr1);
                            getFloatArr(numVal, arr2);
                            getFloatArr(numVal, arr3);
                            break;
                        }
                    }
                    else
                    {
                        switch (dim_)
                        {
                        case 1:
                            skipFloat(numVal);
                            break;
                        case 3:
                            skipFloat(numVal);
                            skipFloat(numVal);
                            skipFloat(numVal);
                            break;
                        }
                    }
                }
            }
        }
    }
    //buildParts(true);

    createDataOutObj(dim, outObjects, baseName, timeStep, numTimeSteps);
}

//
// Destructor
//
DataFileGoldBin::~DataFileGoldBin()
{
}
