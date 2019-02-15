/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2003 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class DataFileGold                    ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 07.04.2003                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "DataFileGold.h"
#include "ReadEnsight.h"

#if defined(__GNUC__) && (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ > 2)
#pragma GCC diagnostic warning "-Wuninitialized"
#endif

//
// Constructor
//
DataFileGold::DataFileGold(const coModule *mod, const string &name, const int &dim, const int &numVals)
    : EnFile(mod, name, dim)
    , lineCnt_(0)
    , numVals_(numVals)
    , indexMap_(NULL)
    , actPartIndex_(0)
{
    className_ = string("DataFileGold");

    if (numVals_ > 0)
    {
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
}

void
DataFileGold::readCells(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep, int numTimeSteps)
{
    if (isOpen_)
    {
        char buf[lineLen + 1];
        // 1 lines decription - ignore it
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        int entries(0);
        float val;
        size_t id(0);
        int actPartNr;
        int numVal = 0;
        EnPart *actPart(NULL);
        float *arr1 = NULL, *arr2 = NULL, *arr3 = NULL;
        int j, eleCnt2d = 0, eleCnt3d = 0;
        float **tArr = NULL;

        while (!feof(in_))
        {
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            string tmp(buf);

            // we found a part line
            id = tmp.find("part");
            if (id != string::npos)
            {
                // part line found   -- we skip it for the moment --
                // length of "part" +1
                fgets(buf, lineLen, in_);
                ++lineCnt_;
                actPartNr = atoi(buf);
                actPart = findPart(actPartNr);
                eleCnt2d = 0, eleCnt3d = 0;
                // allocate memory for whole parts

                if (actPart != NULL)
                {
                    if (actPart->isActive())
                    {
                        //int numCells( actPart->getTotNumEle() );
                        switch (dim_)
                        {
                        case 1:
                            // create mem for 2d/3d data
                            if (actPart->numEleRead2d() > 0)
                                actPart->d2dx_ = new float[actPart->numEleRead2d() + 1];
                            else
                                actPart->d2dx_ = NULL;
                            actPart->d2dy_ = NULL;
                            actPart->d2dz_ = NULL;

                            if (actPart->numEleRead3d() > 0)
                                actPart->d3dx_ = new float[actPart->numEleRead3d() + 1];
                            else
                                actPart->d3dx_ = NULL;
                            actPart->d3dy_ = NULL;
                            actPart->d3dz_ = NULL;
                            //  			    cerr << "DataFileAsc::readCells(..) allocating memory for part "
                            //  				 << actPartNr
                            //  				 << "  "
                            //  				 << actPart->numEleRead3d()
                            //  				 << "  " << actPart->numEleRead2d()<< endl;

                            break;
                        case 3:
                            arr1 = NULL, arr2 = NULL, arr3 = NULL;
                            // create mem for 2d/3d data
                            if (actPart->numEleRead2d() > 0)
                            {
                                arr1 = new float[actPart->numEleRead2d() + 1];
                                arr2 = new float[actPart->numEleRead2d() + 1];
                                arr3 = new float[actPart->numEleRead2d() + 1];
                            }
                            actPart->d2dx_ = arr1;
                            actPart->d2dy_ = arr2;
                            actPart->d2dz_ = arr3;

                            arr1 = NULL, arr2 = NULL, arr3 = NULL;
                            if (actPart->numEleRead3d() > 0)
                            {
                                arr1 = new float[actPart->numEleRead3d()];
                                arr2 = new float[actPart->numEleRead3d()];
                                arr3 = new float[actPart->numEleRead3d()];
                            }
                            actPart->d3dx_ = arr1;
                            actPart->d3dy_ = arr2;
                            actPart->d3dz_ = arr3;
                            break;
                        }
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
                resetPart(actPartNr, actPart);
            }

            // coordinates -line
            id = 0;
            id = tmp.find("coordinates");
            if (id == string::npos)
            {
                if ((actPart != NULL) && (actPart->isActive()))
                {
                    string elementType(strip(tmp));
                    EnElement elem(elementType);

                    int anzEle(0);
                    // we have a valid ENSIGHT element
                    if (elem.valid())
                    {
                        anzEle = actPart->getElementNum(elementType);
                        //                         cerr << "DataFileGoldBin::readCells() " << anzEle << " for " << elementType  << endl;
                        EnElement thisEle = actPart->findElement(elementType);
                        vector<int> bl(thisEle.getBlacklist());
                        if (thisEle.getBlacklist().size() != anzEle)
                        {
                            cerr << "DataFileGold::readCells( ) blacklist size problem " << bl.size() << endl;
                        }

                        int i;
                        switch (dim_)
                        {
                        case 1:
                            // scalar data
                            if (thisEle.getDim() == EnElement::D2)
                            {
                                for (i = 0; i < anzEle; ++i)
                                {
                                    fgets(buf, lineLen, in_);
                                    ++lineCnt_;
                                    entries = sscanf(buf, "%e", &val);
                                    if (bl[i] > 0)
                                    {
                                        actPart->d2dx_[eleCnt2d] = val;
                                        ++eleCnt2d;
                                    }
                                }
                            }
                            else if (thisEle.getDim() == EnElement::D3)
                            {
                                for (i = 0; i < anzEle; ++i)
                                {
                                    fgets(buf, lineLen, in_);
                                    ++lineCnt_;
                                    entries = sscanf(buf, "%e", &val);
                                    if (bl[i] > 0)
                                    {
                                        actPart->d3dx_[eleCnt3d] = val;
                                        ++eleCnt3d;
                                    }
                                }
                            }
                            break;
                        case 3:
                            tArr = new float *[3];
                            for (j = 0; j < 3; j++)
                            {
                                tArr[j] = new float[anzEle];
                                for (i = 0; i < anzEle; ++i)
                                {
                                    fgets(buf, lineLen, in_);
                                    ++lineCnt_;
                                    entries = sscanf(buf, "%e", &(tArr[j][i]));
                                }
                            }
                            if (thisEle.getDim() == EnElement::D2)
                            {
                                for (i = 0; i < anzEle; ++i)
                                {
                                    if (bl[i] > 0)
                                    {
                                        actPart->d2dx_[eleCnt2d] = tArr[0][i];
                                        actPart->d2dy_[eleCnt2d] = tArr[1][i];
                                        actPart->d2dz_[eleCnt2d] = tArr[2][i];
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
                                        actPart->d3dx_[eleCnt3d] = tArr[0][i];
                                        actPart->d3dy_[eleCnt3d] = tArr[1][i];
                                        actPart->d3dz_[eleCnt3d] = tArr[2][i];
                                        ++eleCnt3d;
                                    }
                                }
                            }
                            for (i = 0; i < 3; ++i)
                            {
                                delete[] tArr[i];
                            }
                            delete[] tArr;
                            break;
                        }
                    } // if( elem.valid()

                } // if ( actPart->isActive()
                else
                {
                    EnPart currPart;
                    if (actPart == NULL)
                    {
                        // Actually, actPart should never be NULL since all (even deactivated) parts were added to the partList.
                        currPart = findMasterPart(actPartNr);
                        if (currPart.getPartNum() == 0)
                            cerr << "DataFileGold::readCells() part with number " << actPartNr << "was not found in the master-part-list" << endl << "serious ERROR" << endl;
                    }
                    else
                    {
                        currPart = *actPart;
                    }
                    int numParts = currPart.getNumEle();
                    // skip data
                    while (numParts > 0)
                    {
                        fgets(buf, lineLen, in_);
                        ++lineCnt_;
                        string elementType(strip(buf));
                        EnElement elem(elementType);
                        int anzEle(0);
                        // we have a valid ENSIGHT element
                        if (elem.valid())
                        {
                            anzEle = currPart.getElementNum(elementType);
                            numVal = anzEle * dim_;
                            fgets(buf, lineLen, in_);
                            ++lineCnt_;
                        }
                        numParts--;
                    }
                }
            }
        }
    }
    createDataOutObj(ens, dim, outObjects, baseName, timeStep,false);
}
//
// Method
//
void
DataFileGold::read(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep, int numTimeSteps)
{
    //cerr << "DataFileGold::read() called" << endl;

    if (isOpen_)
    {
        char buf[lineLen];
        // 1 lines decription - ignore it
        fgets(buf, lineLen, in_);
        ++lineCnt_;

        int entries(0);
        float val;
        int numGot(0);
        size_t id(0);
        int actPartNr;
        int numVal = 0;
        EnPart *actPart = NULL, altPart;

        float *arr1 = NULL, *arr2 = NULL, *arr3 = NULL;

        while (!feof(in_))
        {

            fgets(buf, lineLen, in_);
            ++lineCnt_;
            string tmp(buf);

            // we found a part line
            id = tmp.find("part");
            if (id != string::npos)
            {
                // part line found   -- we skip it for the moment --
                fgets(buf, lineLen, in_);
                ++lineCnt_;
                actPartNr = atoi(buf);
                //cerr << "DataFileGold::read( ) part " <<  actPartNr << endl;
                actPart = findPart(actPartNr);
                numGot = 0;

                // allocate memory for whole parts
                if (actPart != NULL)
                {
                    int numVal = actPart->numCoords();
                    if (actPart->isActive())
                    {
                        switch (dim_)
                        {
                        case 1:
                            arr1 = new float[numVal];
                            if (actPart->arr1_ != NULL)
                                delete[] actPart -> arr1_;
                            actPart->arr1_ = arr1;
                            break;
                        case 3:
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
                        cerr << "DataFileGold::read() part with number " << actPartNr << "was not found in the master-part-list" << endl << "serious ERROR" << endl;
                }
            }

            // coordinates -line
            id = 0;
            id = tmp.find("coordinates");
            if (id != string::npos)
            {
                int i(0);
                if ((actPart != NULL) && (actPart->isActive()))
                {
                    numGot = 0;
                    switch (dim_)
                    {
                    case 1:
                        for (i = 0; i < numVal; ++i)
                        {
                            // we have data
                            fgets(buf, lineLen, in_);
                            ++lineCnt_;
                            entries = sscanf(buf, "%e", &val);
                            arr1[numGot] = val;
                            numGot++;
                        }
                        break;
                    case 3:
                        for (i = 0; i < numVal; ++i)
                        {
                            // we have data
                            fgets(buf, lineLen, in_);
                            ++lineCnt_;
                            entries = sscanf(buf, "%e", &val);
                            arr1[numGot] = val;
                            numGot++;
                        }
                        numGot = 0;
                        for (i = 0; i < numVal; ++i)
                        {
                            // we have data
                            fgets(buf, lineLen, in_);
                            ++lineCnt_;
                            entries = sscanf(buf, "%e", &val);
                            arr2[numGot] = val;
                            numGot++;
                        }
                        numGot = 0;
                        numGot = 0;
                        for (i = 0; i < numVal; ++i)
                        {
                            // we have data
                            fgets(buf, lineLen, in_);
                            ++lineCnt_;
                            entries = sscanf(buf, "%e", &val);
                            arr3[numGot] = val;
                            numGot++;
                        }
                        break;
                    }
                }
                else
                {
                    numVal *= dim_;
                    for (i = 0; i < numVal; ++i)
                    {
                        fgets(buf, lineLen, in_);
                        ++lineCnt_;
                    }
                }
            }
        }
    }
    createDataOutObj(ens, dim, outObjects, baseName, timeStep, numTimeSteps);
}

//
// Destructor
//
DataFileGold::~DataFileGold()
{
}
