/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                           (C)2002 / 2003 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class EnGoldgeoBIN                     ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 08.04.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "EnGoldGeoBIN.h"
#include "GeoFileAsc.h"
#include <api/coModule.h>
#include <util/byteswap.h>

#include <vector>

// #define DEBUG

//
// Constructor
//
EnGoldGeoBIN::EnGoldGeoBIN(const coModule *mod)
    : EnFile(mod)
    , lineCnt_(0)
    , numCoords_(0)
    , indexMap_(NULL)
    , maxIndex_(0)
    , lastNc_(0)
    , globalCoordIndexOffset_(0)
    , currElementIdx_(0)
    , currCornerIdx_(0)

{
    className_ = string("EnGoldGeoBIN");
}

EnGoldGeoBIN::EnGoldGeoBIN(const coModule *mod, const string &name, EnFile::BinType binType)
    : EnFile(mod, binType)
    , lineCnt_(0)
    , numCoords_(0)
    , indexMap_(NULL)
    , maxIndex_(0)
    , lastNc_(0)
    , globalCoordIndexOffset_(0)
    , currElementIdx_(0)
    , currCornerIdx_(0)
{
    className_ = string("EnGoldGeoBIN");

#ifdef WIN32
    in_ = fopen(name.c_str(), "rb");
#else
    in_ = fopen(name.c_str(), "r");
#endif
    if (in_)
        isOpen_ = true;
    else
        cerr << className_ << "::EnGoldGeoBIN(..) open NOT successful" << endl;

    //byteSwap_ =  machineIsLittleEndian();
    byteSwap_ = false;

#ifdef DEBUG
    //    byteSwap_ = machineIsLittleEndian();
    //    if ( byteSwap_ ) coModule::sendInfo("%s","Attention:  byte swap needed!" );
    cout << "EnGoldGeoBin constructor called....byteSwap_ = " << ((byteSwap_ == true) ? "true" : "false") << endl;
#endif
}

//
// thats the gereral read method
//
void
EnGoldGeoBIN::read(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects2d, coDistributedObject **outObjects3d, const string &actObjNm2d, const string &actObjNm3d, int &timeStep)
{
    //cerr << className_ << "::read() called" << endl;
    module_->sendInfo("%s", "start reading parts  -  please be patient..");
    // read header
    readHeader();

    // read bounding box
    readBB();

    // allocate memory for coords, connectivities...
    //allocateMemory();

    int allPartsToRead(0);
    if (!masterPL_.empty())
    {
        unsigned int ii;
        for (ii = 0; ii < masterPL_.size(); ++ii)
        {
            if (masterPL_[ii].isActive())
            {
                allPartsToRead++;
            }
        }
    }

    char buf[256];
    int cnt = 0;
    partFound = false;
    // read parts and connectivity
    PartList::iterator it(masterPL_.begin());
    for (; it != masterPL_.end(); it++)
    {
        if (it->isActive())
        {
            EnPart part;
            readPart(part);
            readPartConn(part);
            sprintf(buf, "read part#%d :  %d of %d", it->getPartNum(), cnt, allPartsToRead);
            module_->sendInfo("%s", buf);
            cnt++;
        }
        else
        {
            skipPart();
        }
        globalCoordIndexOffset_ = numCoords_;
    }

    sprintf(buf, "done reading parts  %d parts read", cnt);
    module_->sendInfo("%s", buf);


    createGeoOutObj(ens, dim, outObjects2d, outObjects3d, actObjNm2d, actObjNm3d, timeStep);

    return;
}

// get Bounding Box section in ENSIGHT GOLD (only)
int
EnGoldGeoBIN::readBB()
{
    if (isOpen_)
    {
        // 2 lines decription - ignore it
        string line(getStr());
        if (line.find("extents") != string::npos)
        {
            //cerr <<  className_ << "::readBB()  extents section found" << endl;
            // covise is not using the bounding box so far
            skipFloat(6);
        }
        else
        {
            if (binType_ == EnFile::FBIN)
            {
#ifdef WIN32
                _fseeki64(in_, -88, SEEK_CUR); // 4 + 80 + 4
#else
                fseek(in_, -88, SEEK_CUR); // 4 + 80 + 4
#endif
            }
            else
            {
#ifdef WIN32
                _fseeki64(in_, -80, SEEK_CUR);
#else
                fseek(in_, -80, SEEK_CUR);
#endif
            }
        }
    }
    return 0;
}

// read header
int
EnGoldGeoBIN::readHeader()
{
    int ret(0);
    if (isOpen_)
    {
        // bin type
        getStr();
        // 2 lines decription - ignore it or
        // we may have a multiple timestep - single file situation
        // we ignore it
        string checkTs(getStr());
        size_t tt(checkTs.find("BEGIN TIME STEP"));
        if (tt != string::npos)
        {
            module_->sendInfo("%s", "found multiple timesteps in one file - ONLY THE FIRST TIMESTEP IS READ - ALL OTHERS ARE IGNORED");
            getStr();
        }

        getStr();
        // node id
        string tmp(getStr());
        size_t beg(tmp.find(" id"));

        if (beg == string::npos)
        {
            cerr << className_ << "::readHeader() ERROR node-id not found" << endl;
            return ret;
        }
        // " id"  has length 3
        beg += 4;
        string cmpStr(strip(tmp.substr(beg, 10)));
        nodeId_ = -1; ////////
        if (cmpStr == string("off"))
            nodeId_ = EnFile::OFF;
        else if (cmpStr == string("given"))
            nodeId_ = EnFile::GIVEN;
        else if (cmpStr == string("assign"))
            nodeId_ = EnFile::ASSIGN;
        else if (cmpStr == string("ignore"))
            nodeId_ = EnFile::EN_IGNORE;
        // element id
        tmp = getStr();

        beg = tmp.find(" id");

        if (beg == string::npos)
        {
            cerr << className_ << "::readHeader() ERROR element-id not found" << endl;
            return ret;
        }

        beg += 4;
        cmpStr = strip(tmp.substr(beg, 10));
        if (cmpStr == string("off"))
            elementId_ = EnFile::OFF;
        else if (cmpStr == string("given"))
            elementId_ = EnFile::GIVEN;
        else if (cmpStr == string("assign"))
            elementId_ = EnFile::ASSIGN;
        else if (cmpStr == string("ignore"))
            elementId_ = EnFile::EN_IGNORE;
        else
            cerr << className_ << "::readHeader()  ELEMENT-ID Error" << endl;

        ret = 1;
    }
    return ret;
}

int
EnGoldGeoBIN::readPart(EnPart &actPart)
{
    int ret(0);

    if (isOpen_)
    {
        // 2 lines decription - ignore it

        string line;
        if (!partFound)
        {
            line = getStr();
        }
        if (partFound || line.find("part") != string::npos)
        {
            // part No

            actPartNumber_ = getInt();

            if (actPartNumber_ > 10000 || actPartNumber_ < 0)
            {
                byteSwap_ = !byteSwap_;
                byteSwap(actPartNumber_);
            }

#ifdef DEBUG
            cout << "readPart called....byteSwap_ = " << ((byteSwap_ == true) ? "true" : "false") << endl;
#endif

            actPart.setPartNum(actPartNumber_);
            //cerr << className_ << "::readPart() got part No: " << actPartNumber_ << endl;

            // description line
            actPart.setComment(getStr());
            // coordinates token
            line = getStr();
            if (line.find("coordinates") == string::npos)
            {
                if (line.find("block") != string::npos)
                {
                    module_->sendInfo("%s", "found structured part - not implemented yet -");
                    return -1;
                }

                cerr << className_ << "::readPart() coordinates key not found" << endl;
                return -1;
            }
            // number of coordinates
            int nc(getInt());
            numCoords_ += nc;
            // we don't allocate indexMap_
            // fillIndexMap will do everything for us
            // id's or coordinates
            int i;
            int *iMap(NULL);

            float *x = new float[nc];
            float *y = new float[nc];
            float *z = new float[nc];

            switch (nodeId_)
            {
            case GIVEN:
            {
                // index array
                iMap = new int[nc];
                getIntArr(nc, iMap);
                // workaround for broken EnsightFiles with Index map as floats
                float *tmpf = (float *)(iMap);
                if (nc > 2 && tmpf[0] == 1.0 && tmpf[1] == 2.0 && tmpf[2] == 3.0)
                {
                    fprintf(stderr, "Broken Ensight File!!! ignoring index map\n");
                }
                else
                {
                    for (i = 0; i < nc; ++i)
                    {
                        fillIndexMap(iMap[i], i);
                    }
                }
                // read x-values
                getFloatArr(nc, x);
                // read y-values
                getFloatArr(nc, y);
                // read z-values
                getFloatArr(nc, z);
                delete[] iMap;
            }
            break;
            default:
                // read x-values
                getFloatArr(nc, x);
                // read y-values
                getFloatArr(nc, y);
                // read z-values
                getFloatArr(nc, z);
            }
            actPart.x3d_ = x;
            actPart.y3d_ = y;
            actPart.z3d_ = z;
            actPart.setNumCoords(nc);
        }
        else
            cerr << className_ << "::readPart() "
                 << " NO part header found" << endl;
        //cerr << className_ << "::readPart()  got " << numCoords_ << " coordinates"   << endl;
    }
    return ret;
}

int
EnGoldGeoBIN::readPartConn(EnPart &actPart)
{
#ifdef DEBUG
    cerr << "readPartConn()" << endl;
#endif

    int &partNo(actPartNumber_);
    int ret(0);
    if (!isOpen_)
        return -1;

    char buf[lineLen];

    int currElePtr2d = 0, currElePtr3d = 0;
    int *locArr(NULL);
    int cornIn[20], cornOut[20];
    int numElements, nc, covType;

    partFound = false;
    int numDistCorn;
    int statistic[30];
    int rstatistic[30][30];
    int ii, jj;
    for (ii = 0; ii < 30; ++ii)
    {
        statistic[ii] = 0;
        for (jj = 0; jj < 9; ++jj)
            rstatistic[ii][jj] = 0;
    }
    int degCells(0);

    vector<int> eleLst2d, eleLst3d, cornLst2d, cornLst3d, typeLst2d, typeLst3d;

    // we don't know a priori how many Ensight elements we can expect here therefore we have to read
    // until we find a new 'part'
    lastNc_ = 0;
    while ((!feof(in_)) && (!partFound))
    {
        string tmp(getStr());
        if (tmp.find("part") != string::npos)
        {
            partFound = true;
        }
        // scan for element type
        string elementType(strip(tmp));
        EnElement elem(elementType);
        // we have a valid ENSIGHT element
        if (elem.valid() && !partFound)
        {
            vector<int> blacklist;
            // get number of elements
            numElements = getInt();
#ifdef DEBUG
            cerr << " read " << numElements << " elements" << endl;
#endif

            if (numElements > 0)
            {
                // skip elements id's
                if (elementId_ == GIVEN)
                    skipInt(numElements);

                // ------------------- NFACED ----------------------
                if (elem.getEnTypeStr() == "nfaced")
                {
                    // Read number of faces/points
                    int *numFaces = new int[numElements];
                    int **numPoints = new int *[numElements];
                    getIntArr(numElements, numFaces);
                    for (int i = 0; i < numElements; ++i)
                    {
                        numPoints[i] = new int[numFaces[i]];
                        getIntArr(numFaces[i], numPoints[i]);
                    }
                    if (includePolyeder_)
                    {
                        // Read elements (VARIANT 1)
                        for (int i = 0; i < numElements; ++i)
                        {
                            typeLst3d.push_back(elem.getCovType());
                            eleLst3d.push_back(currElePtr3d);
                            for (int j = 0; j < numFaces[i]; ++j)
                            {
                                nc = numPoints[i][j];
                                locArr = new int[nc];
                                getIntArr(nc, locArr);
                                for (int k = 0; k < nc; ++k)
                                {
                                    cornLst3d.push_back(locArr[k] - 1);
                                    currElePtr3d++;
                                    if ((k != 0) && (locArr[k] == locArr[0]))
                                    {
                                        // The first point appears twice in the face and would destroy
                                        // our "first-point-again-ends-face"-definition. We explicitly
                                        // start a new face here by adding the point again.
                                        cornLst3d.push_back(locArr[k] - 1);
                                        currElePtr3d++;
                                    }
                                }
                                cornLst3d.push_back(locArr[0] - 1);
                                currElePtr3d++; // add first point again to mark the end of the face
                                delete[] locArr;
                            }
                            currElementIdx_++;
                            blacklist.push_back(1);
                        }
                        //                     // Read elements (VARIANT 2) (WARNING: data doesnt match grid for element-based data)
                        //                     for (int i=0; i<numElements; ++i)
                        //                     {
                        //                         for (int j=0; j<numFaces[i]; ++j)
                        //                         {
                        //                             if (j==0)
                        //                                 typeLst3d.push_back(TYPE_POLYEDER);
                        //                             else
                        //                                 typeLst3d.push_back(TYPE_POLYEDERFACE);
                        //                             eleLst3d.push_back(currElePtr3d);
                        //                             nc = numPoints[i][j];
                        //                             locArr = new int[nc];
                        //                             getIntArr(nc, locArr);
                        //                             for (int k=0; k<nc; ++k) cornLst3d.push_back(locArr[k] - 1);
                        //                             delete [] locArr;
                        //                             currElePtr3d += nc;
                        //                         }
                        //                         currElementIdx_++;
                        //                         blacklist.push_back(1);
                        //                     }
                    }
                    else
                    {
                        for (int i = 0; i < numElements; ++i)
                        {
                            for (int j = 0; j < numFaces[i]; ++j)
                            {
                                skipInt(numPoints[i][j]);
                            }
                            blacklist.push_back(-1); // dont read data
                        }
                    }
                    delete[] numFaces;
                    for (int i = 0; i < numElements; ++i)
                        delete[] numPoints[i];
                    delete[] numPoints;
                }
                // ------------------- NFACED ----------------------

                // ------------------- NSIDED ----------------------
                else if (elem.getEnTypeStr() == "nsided")
                {
                    // Read number of points
                    int *numPoints = new int[numElements];
                    getIntArr(numElements, numPoints);
                    // Read elements
                    for (int i = 0; i < numElements; ++i)
                    {
                        typeLst2d.push_back(elem.getCovType());
                        eleLst2d.push_back(currElePtr2d);
                        nc = numPoints[i];
                        locArr = new int[nc];
                        getIntArr(nc, locArr);
                        for (int k = 0; k < nc; ++k)
                        {
                            cornLst2d.push_back(locArr[k] - 1);
                        }
                        delete[] locArr;
                        currElePtr2d += nc;
                        currElementIdx_++;
                        blacklist.push_back(1);
                    }
                    delete[] numPoints;
                }
                // ------------------- NSIDED ----------------------

                // ---------------- DEFAULT ELEMENT-----------------
                else
                {
                    nc = elem.getNumberOfCorners();
                    covType = elem.getCovType();
                    locArr = new int[numElements * nc];
                    getIntArr(numElements * nc, locArr);
                    int eleCnt(0), idx;
                    for (int i = 0; i < numElements; ++i)
                    {
                        // remap indicees (Ensight elements may have a different numbering scheme
                        //                 as COVISE elements)
                        //  prepare arrays
                        int j;
                        for (j = 0; j < nc; ++j)
                        {
                            idx = eleCnt + j;
                            cornIn[j] = locArr[idx] - 1;
                        }
                        eleCnt += nc;
                        // we add the element to the list of points if it has more than one
                        // distinct point
                        numDistCorn = elem.distinctCorners(cornIn, cornOut);
                        if (numDistCorn > 1)
                        {
                            if (numDistCorn != elem.getNumberOfCorners())
                            {
                                int iii = elem.getNumberOfCorners();
                                statistic[iii]++;
                                rstatistic[iii][numDistCorn]++;
                            }
                            // do the remapping
                            elem.remap(cornIn, cornOut);
                            if (elem.getDim() == EnElement::D2)
                            {
                                eleLst2d.push_back(currElePtr2d);
                                for (j = 0; j < nc; ++j)
                                    cornLst2d.push_back(cornOut[j]);
                                typeLst2d.push_back(covType);
                                currElePtr2d += nc;
                            }
                            else if (elem.getDim() == EnElement::D3)
                            {
                                eleLst3d.push_back(currElePtr3d);
                                for (j = 0; j < nc; ++j)
                                    cornLst3d.push_back(cornOut[j]);
                                typeLst3d.push_back(covType);
                                currElePtr3d += nc;
                            }
                            currElementIdx_++;
                            blacklist.push_back(1);
                        }
                        else
                        {
                            blacklist.push_back(-1);
                            degCells++;
                        }
                    }
                    delete[] locArr;
                    lastNc_ = nc;
                }
                // ---------------- DEFAULT ELEMENT-----------------
            }

            elem.setBlacklist(blacklist);
            actPart.addElement(elem, numElements);
        }
    }
    // we have read one line more than needed
    if (partFound)
    {
        //	in_.sync();
        //	if ( binType_ == EnFile::FBIN) in_.seekg(-88L, ios::cur);
        //	else in_.seekg(-80L, ios::cur);
        //	in_.sync();
    }

    if (partList_ != NULL)
    {
        // create arrys explicitly
        int *elePtr2d(NULL), *typePtr2d(NULL), *connPtr2d(NULL);
        int *elePtr3d(NULL), *typePtr3d(NULL), *connPtr3d(NULL);

        elePtr2d = new int[eleLst2d.size()];
        elePtr3d = new int[eleLst3d.size()];
        typePtr2d = new int[typeLst2d.size()];
        typePtr3d = new int[typeLst3d.size()];
        connPtr2d = new int[cornLst2d.size()];
        connPtr3d = new int[cornLst3d.size()];

        std::copy(eleLst2d.begin(), eleLst2d.end(), elePtr2d);
        std::copy(eleLst3d.begin(), eleLst3d.end(), elePtr3d);
        std::copy(typeLst2d.begin(), typeLst2d.end(), typePtr2d);
        std::copy(typeLst3d.begin(), typeLst3d.end(), typePtr3d);
        std::copy(cornLst2d.begin(), cornLst2d.end(), connPtr2d);
        std::copy(cornLst3d.begin(), cornLst3d.end(), connPtr3d);
        actPart.setNumEleRead2d(eleLst2d.size());
        actPart.setNumEleRead3d(eleLst3d.size());
        actPart.setNumConnRead2d(cornLst2d.size());
        actPart.setNumConnRead3d(cornLst3d.size());
        actPart.el2d_ = elePtr2d;
        actPart.tl2d_ = typePtr2d;
        actPart.cl2d_ = connPtr2d;
        actPart.el3d_ = elePtr3d;
        actPart.tl3d_ = typePtr3d;
        actPart.cl3d_ = connPtr3d;

        partList_->push_back(actPart);
    }

    if (degCells > 0)
    {
        cerr << " WRONG ELEMENT STATISTICS" << endl;
        cerr << "-------------------------------------------" << endl;
        for (ii = 2; ii < 9; ++ii)
        {
            cerr << ii << " | " << statistic[ii];
            for (jj = 1; jj < 9; ++jj)
                cerr << " || " << rstatistic[ii][jj];
            cerr << endl;
        }

        sprintf(buf, " -> found %d fully degenerated cells in part %d", degCells, partNo);
        module_->sendInfo("%s", buf);
    }

    return ret;
}

//
// Destructor
//
EnGoldGeoBIN::~EnGoldGeoBIN()
{
}

//
// set up a list of parts
//
void
EnGoldGeoBIN::parseForParts()
{
    int numParts(0);
    int totNumElements;

    EnPart *actPart(NULL);

    readHeader();
    readBB();
    module_->sendInfo("%s", "getting parts  -  please wait...");

	if (!isOpen_)
	{
		return;
	}
    int cnt = 0;
    bool validElementFound = false;
    while (!feof(in_))
    {
        string tmp(getStr());
        int actPartNr;
        // scan for part token
        // read comment and print part line
        size_t id = tmp.find("part");
        if (id != string::npos)
        {
            // part line found
            // get part number
            actPartNr = getInt();
            if (actPartNr > 10000 || actPartNr < 0)
            {
                byteSwap_ = !byteSwap_;
                byteSwap(actPartNr);
            }

#ifdef DEBUG
            cout << "parseForParts called....byteSwap_ = " << ((byteSwap_ == true) ? "true" : "false") << endl;
#endif

            // comment line we need it for the table output
            string comment(getStr());
            // coordinates token
            string coordTok(getStr());
            id = coordTok.find("coordinates");
            int numCoords(0);
            if (id != string::npos)
            {
                // number of coordinates
                numCoords = getInt();
                switch (nodeId_)
                {
                case GIVEN:
                    skipInt(numCoords);
                    skipFloat(numCoords);
                    skipFloat(numCoords);
                    skipFloat(numCoords);
                    break;
                default:
                    skipFloat(numCoords);
                    skipFloat(numCoords);
                    skipFloat(numCoords);
                }
            }
            id = string::npos;
            // add part to the partList
            // BE AWARE that you have to add a part to the list of parts AFTER the part is
            // COMPLETELY build up
            if (partList_ != NULL)
            {
                if (actPart != NULL)
                {
                    partList_->push_back(*actPart);
                    delete actPart;
                }
            }
            actPart = new EnPart(actPartNr, comment);
            actPart->setNumCoords(numCoords);

            // now increment number of parts and reset element counter
            numParts++;
            totNumElements = 0;
        }
        // scan for element type
        string elementType(strip(tmp));
        EnElement elem(elementType);
        // we have a valid ENSIGHT element
        if (actPart->comment().find("particles") != string::npos)
        {
            validElementFound = true;
        }
        else if (elem.valid())
        {
            // get number of elements
            int numElements = getInt();
            if (elementId_ == GIVEN)
                skipInt(numElements);

            if (elem.getEnTypeStr() == "nfaced")
            {
                int numFaces = 0;
                int numNodes = 0;
                for (int i = 0; i < numElements; i++)
                    numFaces += getInt();
                for (int i = 0; i < numFaces; i++)
                    numNodes += getInt();
                skipInt(numNodes);
            }
            else if (elem.getEnTypeStr() == "nsided")
            {
                int numPoints = 0;
                for (int i = 0; i < numElements; i++)
                    numPoints += getInt();
                skipInt(numPoints);
            }
            else
            {
                skipInt(elem.getNumberOfCorners() * numElements);
            }

            // we set the number of 2/3D elements to 1 if
            // we find the cooresponding element type that's suficient to mark
            // the dimension of the current part. The real numbers will be set
            // during the read phase
            if (elem.getDim() == EnElement::D2)
            {
                if (actPart != NULL)
                    actPart->setNumEleRead2d(1);
            }
            if (elem.getDim() == EnElement::D3)
            {
                if (actPart != NULL)
                    actPart->setNumEleRead3d(1);
            }

            // add element info to the part
            if (actPart != NULL)
                actPart->addElement(elem, numElements);
            totNumElements += numElements;
            validElementFound = true;
        } // if( elem.valid() )
        cnt++;
    }

    // add last part to the list of parts
    if (partList_ != NULL)
    {
        if (actPart != NULL)
        {
            partList_->push_back(*actPart);
            delete actPart;
        }
    }
    else
    {
        cerr << className_ << "::parseForParts() WARNING partList_ NULL" << endl;
    }

    if (!validElementFound)
    {
        cerr << className_ << "::parseForParts() WARNING never found a valid element SUSPICIOUS!!!!" << endl;
        fileMayBeCorrupt_ = true;
        return;
    }

    sendPartsToInfo();
}

int
EnGoldGeoBIN::allocateMemory()
{
    int totNumCoords(0);
    int totNumEle(0);
    int totNumCorners(0);

    if (!masterPL_.empty())
    {
        unsigned int ii;
        for (ii = 0; ii < masterPL_.size(); ++ii)
        {
            if (masterPL_[ii].isActive())
            {
                totNumEle += masterPL_[ii].getTotNumEle();
                totNumCorners += masterPL_[ii].getTotNumberOfCorners();
                totNumCoords += masterPL_[ii].numCoords();
            }
        }
        // prepare dc - allocate fields
        dc_.setNumCoord(totNumCoords);
        dc_.x = new float[totNumCoords];
        dc_.y = new float[totNumCoords];
        dc_.z = new float[totNumCoords];

        dc_.setNumElem(totNumEle);
        dc_.el = new int[totNumEle];
        dc_.tl = new int[totNumEle];

        dc_.cl = new int[totNumCorners];
        dc_.setNumConn(totNumCorners);

        allocated_ = (dc_.x != NULL) && (dc_.y != NULL) && (dc_.z != NULL);
        allocated_ &= (dc_.el != NULL);
        allocated_ &= (dc_.tl != NULL);

        if (!allocated_)
            cerr << className_ << "::allocateMemory() ALLOCATION FAILED" << endl;

        // 	cerr << className_ << "::allocateMemory() " << totNumEle << " elements, "
        // 	     << totNumCorners << " corners" << " and " << totNumCoords << " coordinates" << endl;
    }
    else
    {
        cerr << className_ << "::readConn() masterPL_ is empty() " << endl;
    }
    return 0;
}

int
EnGoldGeoBIN::skipPart()
{
#ifdef DEBUG
    cerr << "skipPart()" << endl;
#endif

    // Even in skipPart we add the part (with basic information) to the partList but deactivate it.
    // We need it for skipping the corresponding data in case of "really" transient grids.
    EnPart actPart;
    actPart.activate(false);

    size_t id;
    string line;
    if (!partFound)
    {
        line = getStr();
    }
    if (partFound || line.find("part") != string::npos)
    {
        // part line found
        // get part number
        int pn = getInt();
        if (pn > 10000 || pn < 0)
        {
            byteSwap_ = !byteSwap_;
            byteSwap(pn);
        }
        actPart.setPartNum(pn);

#ifdef DEBUG
        cout << "skipPart called....byteSwap_ = " << ((byteSwap_ == true) ? "true" : "false") << endl;
#endif

        // description line
        actPart.setComment(getStr());

        // coordinates token
        string coordTok(getStr());
        // int id = coordTok.find("coordinates");
        id = coordTok.find("coordinates");
        int numCoords(0);
        if (id != string::npos)
        {
            // number of coordinates
            numCoords = getInt();
            actPart.setNumCoords(numCoords);
            //cerr << className_ << "::skipPart() numCoords " << numCoords << endl;
            switch (nodeId_)
            {
            case GIVEN:
                skipInt(numCoords);
                skipFloat(3 * numCoords);
                break;
            default:
                skipFloat(3 * numCoords); // Warning:  this won't work if 3*numCoords*4(bytes) is bigger than 4GB
            }
        }
    }
    // scan for element type

    while (!feof(in_))
    {
        string tmp(getStr());

        // scan for part token
        // int id = tmp.find("part");
        id = tmp.find("part");
        // part found - rewind one line and exit
        if (id != string::npos)
        {
            if (partList_ != NULL)
                partList_->push_back(actPart);
            if (binType_ == EnFile::FBIN)
            {
#ifdef WIN32
                _fseeki64(in_, -88L, SEEK_CUR);
#else
                fseek(in_, -88L, SEEK_CUR);
#endif
            }
            else
            {
#ifdef WIN32
                _fseeki64(in_, -80L, SEEK_CUR);
#else
                fseek(in_, -80L, SEEK_CUR);
#endif
            }
            partFound = false; // read part string again next time;
            return 0;
        }

        string elementType(strip(tmp));
        EnElement elem(elementType);

        // we have a valid ENSIGHT element
        if (elem.valid())
        {
            // get number of elements
            int numElements = getInt();
            if (elementId_ == GIVEN)
            {
                skipInt(numElements);
            }
            if (elem.getEnTypeStr() == "nfaced")
            {
                int numFaces = 0;
                int numNodes = 0;
                for (int i = 0; i < numElements; i++)
                    numFaces += getInt();
                for (int i = 0; i < numFaces; i++)
                    numNodes += getInt();
                skipInt(numNodes);
            }
            else if (elem.getEnTypeStr() == "nsided")
            {
                int numPoints = 0;
                for (int i = 0; i < numElements; i++)
                    numPoints += getInt();
                skipInt(numPoints);
            }
            else
            {
                skipInt(elem.getNumberOfCorners() * numElements);
            }
            actPart.addElement(elem, numElements);
        } // if( elem.valid() )
    }

    if (partList_ != NULL)
        partList_->push_back(actPart);

    return 0;
}

void
EnGoldGeoBIN::fillIndexMap(const int &i, const int &natIdx)
{
    const int offSet(10000);
    // initial
    if (maxIndex_ == 0)
    {
        maxIndex_ = numCoords_;
        indexMap_ = new int[maxIndex_];
    }
    // realloc
    if (i >= maxIndex_)
    {
        int *tmp = new int[i + offSet];
        int j;
        for (j = 0; j < maxIndex_; ++j)
            tmp[j] = indexMap_[j];
        maxIndex_ = i + offSet;
        delete[] indexMap_;
        indexMap_ = tmp;
    }
    indexMap_[i] = natIdx;
}
