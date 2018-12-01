/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class DataFileAsc                     ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 07.06.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "DataFileAsc.h"
#include "EnElement.h"

#include <util/coviseCompat.h>

//
// Constructor
//
DataFileAsc::DataFileAsc(const coModule *mod)
    : EnFile(mod)
    , lineCnt_(0)
    , numVals_(0)
    , indexMap_(NULL)
    , actPartIndex_(0)
{
    className_ = string("DataFileAsc");
}

DataFileAsc::DataFileAsc(const coModule *mod, const string &name, const int &dim, const int &numVals)
    : EnFile(mod, name, dim)
    , lineCnt_(0)
    , numVals_(numVals)
    , indexMap_(NULL)
    , actPartIndex_(0)
{
    className_ = string("DataFileAsc");

    // if numVals == 0 the memory in tha data container will be allocated
    // dynamically
    if (numVals_ > 0)
    {

        switch (dim_)
        {
        case 1:
            dc_.setNumCoord(numVals_);
            dc_.x = new float[numVals_];
            dc_.y = NULL;
            dc_.z = NULL;
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

void
DataFileAsc::setIndexMap(const int *im)
{
    indexMap_ = const_cast<int *>(im);
}

//
// read data
//
void
DataFileAsc::read(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep)
{
    if (!isOpen_)
        return;
    char buf[lineLen];
    // 1 lines decription - ignore it
    if (!fgets(buf, lineLen, in_))
    {
        cerr << "DataFileAsc: premature EOF 1" << endl;
        return;
    }
    ++lineCnt_;

    int entries(0), numGot(0);
    float val[6];

    while (!feof(in_))
    {
        if (!fgets(buf, lineLen, in_))
        {
            cerr << "DataFileAsc: premature EOF 2" << endl;
            break;
        }
        ++lineCnt_;

        entries = sscanf(buf, "%e%e%e%e%e%e", &val[0], &val[1], &val[2], &val[3], &val[4], &val[5]);

        // TBD: insert error handler
        int i;
        switch (dim_)
        {
        case 1:
            for (i = 0; i < entries; ++i)
            {
                if (numGot < numVals_)
                    dc_.x[numGot] = val[i];
                numGot++;
            }
            break;
        case 3:
            for (i = 0; i < 1 + entries / 2; i += 3)
            {
                if (numGot < numVals_)
                {
                    dc_.x[numGot] = val[i];
                    dc_.y[numGot] = val[i + 1];
                    dc_.z[numGot] = val[i + 2];
                }
                numGot++;
            }
            break;
        }
    }

    createDataOutObj(ens, dim, outObjects, baseName, timeStep);
}

//
// reads cell based data
//
// WARNING / TBD:
// works only if the sequence of cells is identical to the one in the
// corresponding geometry file
//
void
DataFileAsc::readCells(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep)
{
    if (isOpen_)
    {
        char buf[lineLen];
        // 1 lines decription - ignore it
        if (!fgets(buf, lineLen, in_))
        {
            cerr << "DataFileAsc: premature EOF 3" << endl;
            return;
        }
        ++lineCnt_;

        int entries(0);
        float val[6];
        int numGot2d = 0, numGot3d = 0;
        size_t id(0);
        int actPartNr;

        EnPart *actPart = NULL;

        float *arr1, *arr2, *arr3;

        while (!feof(in_))
        {
            if (!fgets(buf, lineLen, in_))
            {
                cerr << "DataFileAsc: premature EOF 4" << endl;
                break;
            }
            ++lineCnt_;
            string tmp(buf);

            bool skip(false);

            id = tmp.find("part");
            if (id != string::npos)
            {
                // part line found   -- we skip it for the moment --
                // length of "part" +1
                string pnumStr(strip(tmp.substr(id + 5)));
                actPartNr = atoi(pnumStr.c_str());
                skip = true;
                actPart = findPart(actPartNr);
                numGot2d = 0;
                numGot3d = 0;
                // allocate memory for whole parts
                if (actPart == NULL)
                {
                    cerr << "DataFileAsc::readCells() part mismatch exiting.. " << lineCnt_ << endl;
                    exit(0);
                }

                if (actPart != NULL)
                {
                    if (actPart->isActive())
                    {
                        //int numCells( actPart->getTotNumEle() );
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
                            // 			    cerr << "DataFileAsc::readCells(..) allocating memory for part "
                            // 				 << actPartNr
                            // 				 << "  "
                            // 				 << numCells
                            // 				 << "  " << actPart->arr1_
                            // 				 << "  " << lineCnt_ << endl;

                            break;
                        case 3:
                            // create mem for 2d/3d data
                            arr1 = new float[actPart->numEleRead2d()];
                            arr2 = new float[actPart->numEleRead2d()];
                            arr3 = new float[actPart->numEleRead2d()];
                            actPart->d2dx_ = arr1;
                            actPart->d2dy_ = arr2;
                            actPart->d2dz_ = arr3;

                            arr1 = new float[actPart->numEleRead3d()];
                            arr2 = new float[actPart->numEleRead3d()];
                            arr3 = new float[actPart->numEleRead3d()];
                            actPart->d3dx_ = arr1;
                            actPart->d3dy_ = arr2;
                            actPart->d3dz_ = arr3;
                            break;
                        }
                    }
                }
                resetPart(actPartNr, actPart);
            }

            // no part line found can be a element identifier like hexa8 penta6...
            string elementType(strip(tmp));
            EnElement elem(actPart->findElement(elementType));
            if (elem.valid())
            {
                int numberOfElements(0);
                if (actPart != NULL)
                    numberOfElements = actPart->getElementNum(elementType);

                if (numberOfElements > 0)
                {
                    int valuesToRead(numberOfElements * dim_);

                    // 		    cerr << "DataFileAsc::readCells(..) read "
                    // 			 << elementType
                    // 			 << " number of values " << valuesToRead << endl;
                    while (valuesToRead > 0)
                    {
                        if (!fgets(buf, lineLen, in_))
                        {
                            cerr << "DataFileAsc: premature EOF 5" << endl;
                            break;
                        }
                        ++lineCnt_;
                        entries = sscanf(buf, "%e%e%e%e%e%e", &val[0], &val[1], &val[2], &val[3], &val[4], &val[5]);

                        valuesToRead -= entries;
                        if (actPart->isActive())
                        {
                            int i;
                            switch (dim_)
                            {
                            case 1:
                                for (i = 0; i < entries; ++i)
                                {
                                    if (elem.getDim() == EnElement::D2)
                                    {
                                        actPart->d2dx_[numGot2d] = val[i];
                                        numGot2d++;
                                    }
                                    if (elem.getDim() == EnElement::D3)
                                    {
                                        actPart->d3dx_[numGot3d] = val[i];
                                        numGot3d++;
                                    }
                                }
                                break;
                            case 3:
                                for (i = 0; i < 1 + entries / 2; i += 3)
                                {
                                    if (elem.getDim() == EnElement::D2)
                                    {
                                        actPart->d2dx_[numGot2d] = val[i];
                                        actPart->d2dy_[numGot2d] = val[i + 1];
                                        actPart->d2dz_[numGot2d] = val[i + 2];
                                        numGot2d++;
                                    }
                                    if (elem.getDim() == EnElement::D3)
                                    {
                                        actPart->d3dx_[numGot3d] = val[i];
                                        actPart->d3dy_[numGot3d] = val[i + 1];
                                        actPart->d3dz_[numGot3d] = val[i + 2];
                                        numGot3d++;
                                    }
                                }
                                break;
                            } // switch
                        } // isActive

                    } // while
                } // if ( numberOfElements > 0 )
            } //if ( elem.valid() )
        }
    }

    createDataOutObj(ens, dim, outObjects, baseName, timeStep,false);
}
