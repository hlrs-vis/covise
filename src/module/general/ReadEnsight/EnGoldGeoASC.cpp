/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                           (C)2002 / 2003 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class EnGoldgeoASC                     ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 08.04.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "EnGoldGeoASC.h"
#include "GeoFileAsc.h"
#include "api/coModule.h"
#include "ReadEnsight.h"

#include <vector>

//
// Constructor
//
EnGoldGeoASC::EnGoldGeoASC(ReadEnsight *mod)
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
    className_ = string("EnGoldGeoASC");
}

EnGoldGeoASC::EnGoldGeoASC(ReadEnsight *mod, const string &name)
    : EnFile(mod, name)
    , lineCnt_(0)
    , numCoords_(0)
    , indexMap_(NULL)
    , maxIndex_(0)
    , lastNc_(0)
    , globalCoordIndexOffset_(0)
    , currElementIdx_(0)
    , currCornerIdx_(0)
{
    className_ = string("EnGoldGeoASC");
}

//
// thats the gereral read method
//
void
EnGoldGeoASC::read(dimType dim, coDistributedObject **outObjects2d, coDistributedObject **outObjects3d, const string &actObjNm2d, const string &actObjNm3d, int &timeStep, int numTimeSteps)
{
    cerr << className_ << "::read() called" << endl;
    ens->sendInfo("%s", "start reading parts  -  please be patient..");
    // read header
    readHeader();

    // read bounding box
    readBB();

    // allocate memory for coords, connectivities...
    allocateMemory();

    int allPartsToRead(0);
	unsigned int ii;
	for (ii = 0; ii < ens->masterPL_.size(); ++ii)
	{
		if (ens->masterPL_[ii].isActive())
		{
			allPartsToRead++;
		}
	}

    char buf[256];
    int cnt(1);
    // read parts and connectivity
    PartList::iterator it(ens->masterPL_.begin());
    for (; it != ens->masterPL_.end(); it++)
    {
        if (it->isActive())
        {
            EnPart part;
            readPart(part);
            readPartConn(part);
            sprintf(buf, "read part#%d :  %d of %d", it->getPartNum(), cnt, allPartsToRead);
            ens->sendInfo("%s", buf);
            cnt++;
        }
        else
        {
            skipPart();
        }
        globalCoordIndexOffset_ = numCoords_;
    }

    createGeoOutObj(dim, outObjects2d, outObjects3d, actObjNm2d, actObjNm3d, timeStep);
    return;
}

// get Bounding Box section in ENSIGHT GOLD (only)
int
EnGoldGeoASC::readBB()
{
    if (isOpen_)
    {
        char buf[lineLen];
        // 2 lines decription - ignore it
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string line(buf);
        if (line.find("extents") != string::npos)
        {
            //cerr <<  className_ << "::readBB()  extents section found" << endl;
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            fgets(buf, lineLen, in_);
            ++lineCnt_;
        }
        // rewind one line
        else
        {
#ifdef WIN32
            _fseeki64(in_, -5L, SEEK_CUR);
#else
            fseek(in_, -5L, SEEK_CUR);
#endif
        }
    }
    return 0;
}

// read header
int
EnGoldGeoASC::readHeader()
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

int
EnGoldGeoASC::readPart(EnPart &actPart)
{
    int ret(0);
    int partNo;

    if (isOpen_)
    {
        char buf[lineLen];
        int noRead(0);
        // 2 lines decription - ignore it
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string line(buf);
        cerr << lineCnt_ << " : " << line << endl;
        if (line.find("part") != string::npos)
        {
            // part No
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            noRead = sscanf(buf, "%d", &partNo);
            actPartNumber_ = partNo;
            actPart.setPartNum(partNo);
            //cerr << className_ << "::readPart() got part No: " << partNo << endl;
            if (noRead != 1)
            {
                cerr << className_ << "::readPart() Error reading part No" << endl;
                return -1;
            }

            // description line
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            actPart.setComment(buf);
            // coordinates token
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            line = string(buf);
            if (line.find("coordinates") == string::npos)
            {
                cerr << className_ << "::readPart() coordinates key not found" << endl;
                return -1;
            }
            // number of coordinates
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            int nc(0);
            noRead = sscanf(buf, "%d", &nc);
            //cerr << className_ << "::readPart() got No. of coordinates (per part): " << nc << endl;
            if (noRead != 1)
            {
                cerr << className_ << "::readPart() Error reading no of coordinates" << endl;
                return -1;
            }
            numCoords_ += nc;
            int numCoords = nc;

            // allocate memory for the coordinate list per part
            float *x = new float[nc];
            float *y = new float[nc];
            float *z = new float[nc];

            // id's or coordinates
            float val;
            int i;
            switch (nodeId_)
            {
            case OFF:
            case ASSIGN:
            case EN_IGNORE:
                // read x-values
                for (i = 0; i < nc; ++i)
                {
                    fgets(buf, lineLen, in_);
                    ++lineCnt_;
                    noRead = sscanf(buf, "%e", &val);
                    if (noRead != 1)
                    {
                        cerr << className_ << "::readPart() Error reading coordinates" << endl;
                        return -1;
                    }
                    x[i] = val;
                }
                // read y-values
                for (i = 0; i < nc; ++i)
                {
                    fgets(buf, lineLen, in_);
                    ++lineCnt_;
                    noRead = sscanf(buf, "%e", &val);
                    if (noRead != 1)
                    {
                        cerr << className_ << "::readPart() Error reading coordinates" << endl;
                        return -1;
                    }
                    y[i] = val;
                }
                // read z-values
                for (i = 0; i < nc; ++i)
                {
                    fgets(buf, lineLen, in_);
                    ++lineCnt_;
                    noRead = sscanf(buf, "%e", &val);
                    if (noRead != 1)
                    {
                        cerr << className_ << "::readPart() Error reading coordinates" << endl;
                        return -1;
                    }
                    z[i] = val;
                }
                break;
            case GIVEN:
                // index array
                // the index array is currently ignored
                int iVal;
                for (i = 0; i < nc; ++i)
                {
                    fgets(buf, lineLen, in_);
                    ++lineCnt_;
                    noRead = sscanf(buf, "%d", &iVal);
                    if (noRead != 1)
                    {
                        cerr << className_ << "::readPart() Error reading index arry" << endl;
                        return -1;
                    }
                    //fillIndexMap(iVal,i);
                }
                // read x-values
                for (i = 0; i < nc; ++i)
                {
                    fgets(buf, lineLen, in_);
                    ++lineCnt_;
                    noRead = sscanf(buf, "%e", &val);
                    if (noRead != 1)
                    {
                        cerr << className_ << "::readPart() Error reading coordinates" << endl;
                        return -1;
                    }
                    x[i] = val;
                }
                // read y-values
                for (i = 0; i < nc; ++i)
                {
                    fgets(buf, lineLen, in_);
                    ++lineCnt_;
                    noRead = sscanf(buf, "%e", &val);
                    if (noRead != 1)
                    {
                        cerr << className_ << "::readPart() Error reading coordinates" << endl;
                        return -1;
                    }
                    y[i] = val;
                }
                // read z-values
                for (i = 0; i < nc; ++i)
                {
                    fgets(buf, lineLen, in_);
                    ++lineCnt_;
                    noRead = sscanf(buf, "%e", &val);
                    if (noRead != 1)
                    {
                        cerr << className_ << "::readPart() Error reading coordinates" << endl;
                        return -1;
                    }
                    z[i] = val;
                }
                break;
            }
            actPart.x3d_ = x;
            actPart.y3d_ = y;
            actPart.z3d_ = z;
            actPart.setNumCoords(numCoords);
        }
        else
            cerr << className_ << "::readPart() " << lineCnt_ << " NO part header found" << endl;
        //cerr << className_ << "::readPart()  got " << numCoords_ << " coordinates"   << endl;
    }
    return ret;
}

int
EnGoldGeoASC::readPartConn(EnPart &actPart)
{

    int &partNo(actPartNumber_);
    int ret(0);

    if (!isOpen_)
        return -1;

    char buf[lineLen];
    int currElePtr3d = 0, currElePtr2d = 0;

    int *locArr = new int[21]; // an ENSIGHT element has max. 20 corners + 1 index
    int cornIn[20], cornOut[20];

    int numElements;
    int nc;
    int covType;
    int numDistCorn(0);

    bool partFound(false);

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
    while ((!feof(in_)) && (!partFound))
    {
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string tmp(buf);

        if (tmp.find("part") != string::npos)
        {
            partFound = true;
            break;
        }
        // scan for element type
        string elementType(strip(tmp));
        EnElement elem(elementType);
        // we have a valid ENSIGHT element
        if (elem.valid())
        {
            vector<int> blacklist;
            // get number of elements
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            numElements = atoi(buf);
            //cerr << className_ << "::readPartConn() " << elementType << "  " <<  numElements <<  endl;
            nc = elem.getNumberOfCorners();
            covType = elem.getCovType();

            int i;
            // read elements id's
            switch (elementId_)
            {
            case OFF:
            case EN_IGNORE:
            case ASSIGN:
                break;
            case GIVEN:
                for (i = 0; i < numElements; ++i)
                {
                    fgets(buf, lineLen, in_);
                    ++lineCnt_;
                }
                break;
            }
            //if (currElementIdx_ == 0) currElePtr = 0;
            //else currElePtr = dc_.el[ currElementIdx_ - 1 ] + lastNc_;
            for (i = 0; i < numElements; ++i)
            {
                fgets(buf, lineLen, in_);
                ++lineCnt_;
                // an integer always has 10 figures (see ENSIGHT docu EnGold)
                En6GeoASC::atoiArr(10, buf, locArr, nc);
                // remap indicees (Ensight elements may have a different numbering scheme
                //                 as COVISE elements)
                //  prepare arrays
                int j;
                for (j = 0; j < nc; ++j)
                    cornIn[j] = locArr[j] - 1;
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
                    // assign element list
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
            lastNc_ = nc;
            elem.setBlacklist(blacklist);
            actPart.addElement(elem, numElements);
        }
    }
    // we have read one line more than needed
    if (partFound)
    {
#ifdef WIN32
        _fseeki64(in_, -5L, SEEK_CUR);
#else
        fseek(in_, -5L, SEEK_CUR);
#endif
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

    delete[] locArr;

    //cerr << className_ << "::readPartConn() " << degCells << " degenerated cells found" << endl;

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
        ens->sendInfo("%s", buf);
    }
    return ret;
}

//
// Destructor
//
EnGoldGeoASC::~EnGoldGeoASC()
{
}

//
// set up a list of parts
//
void
EnGoldGeoASC::parseForParts()
{
    char buf[lineLen];
    int numParts(0);
    int totNumElements;

    EnPart *actPart(NULL);

    readHeader();
    ens->sendInfo("%s", "getting parts  -  please wait...");

    while (!feof(in_))
    {
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string tmp(buf);
        int actPartNr;

        // scan for part token
        // read comment and print part line
        size_t id = tmp.find("part");

        if (id != string::npos)
        {
            // part line found
            // get part number
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            actPartNr = atoi(buf);

            // comment line we need it for the table output
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            string comment(buf);

            // coordinates token
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            string coordTok(buf);
            id = coordTok.find("coordinates");
            int numCoords(0);
            if (id != string::npos)
            {
                // number of coordinates
                fgets(buf, lineLen, in_);
                ++lineCnt_;
                numCoords = atoi(buf);
                //cerr << className_ << "::parseForParts() numCoords " << numCoords << endl;
                switch (nodeId_)
                {
                case GIVEN: // 10+1 + 3*(12+1)
#ifdef WIN32
                    _fseeki64(in_, 50 * numCoords, SEEK_CUR);
#else
                    fseek(in_, 50 * numCoords, SEEK_CUR);
#endif
                    break;
                default: // 39 = 3 * (12 + 1)
#ifdef WIN32
                    _fseeki64(in_, 39 * numCoords, SEEK_CUR);
#else
                    fseek(in_, 39 * numCoords, SEEK_CUR);
#endif
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
        if (elem.valid())
        {
            // get number of elements
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            int numElements = atoi(buf);
            int nc(elem.getNumberOfCorners());
            switch (elementId_)
            {
            case GIVEN:
#ifdef WIN32
                _fseeki64(in_, ((nc * 10) + 12) * numElements, SEEK_CUR);
#else
                fseek(in_, ((nc * 10) + 12) * numElements, SEEK_CUR);
#endif
                break;
            default:
#ifdef WIN32
                _fseeki64(in_, ((nc * 10) + 1) * numElements, SEEK_CUR);
#else
                fseek(in_, ((nc * 10) + 1) * numElements, SEEK_CUR);
#endif
            }

            // add element info to the part
            // we set the number of 2/3D elements to 1 if
            // we find the cooresponding element type that's suuficient to mark
            // the dimension of the cuurent part. The real numbers will be set
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

            if (actPart != NULL)
                actPart->addElement(elem, numElements);

            totNumElements += numElements;

        } // if( elem.valid() )
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
        cerr << "En6GeoASC::parseForParts() WARNING partList_ NULL" << endl;
    }

    sendPartsToInfo();
    //cerr << className_ << "::parseForParts() FINISHED " << endl;
}

int
EnGoldGeoASC::allocateMemory()
{
    int totNumCoords(0);
    int totNumEle(0);
    int totNumCorners(0);

    if (!ens->masterPL_.empty())
    {
        unsigned int ii;
        for (ii = 0; ii < ens->masterPL_.size(); ++ii)
        {
            if (ens->masterPL_[ii].isActive())
            {
                totNumEle += ens->masterPL_[ii].getTotNumEle();
                totNumCorners += ens->masterPL_[ii].getTotNumberOfCorners();
                totNumCoords += ens->masterPL_[ii].numCoords();
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
EnGoldGeoASC::skipPart()
{
    char buf[lineLen];

    // Even in skipPart we add the part (with basic information) to the partList but deactivate it.
    // We need it for skipping the corresponding data in case of "really" transient grids.
    EnPart actPart;
    actPart.activate(false);

    fgets(buf, lineLen, in_);
    ++lineCnt_;
    string tmp(buf);

    // scan for part token
    // read comment and print part line
    size_t id = tmp.find("part");

    if (id != string::npos)
    {
        // part line found
        // get part number
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        int actPartNr = atoi(buf);
        actPart.setPartNum(actPartNr);

        // comment line we need it for the table output
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string comment(buf);
        actPart.setComment(comment);

        // coordinates token
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string coordTok(buf);
        id = coordTok.find("coordinates");
        int numCoords(0);
        if (id != string::npos)
        {
            // number of coordinates
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            numCoords = atoi(buf);
            actPart.setNumCoords(numCoords);
            //cerr << className_ << "::skipPart() numCoords " << numCoords << endl;
            switch (nodeId_)
            {
            case GIVEN:
#ifdef WIN32
                _fseeki64(in_, 50 * numCoords, SEEK_CUR);
#else
                fseek(in_, 50 * numCoords, SEEK_CUR);
#endif

                break;
            default:
#ifdef WIN32
                _fseeki64(in_, 39 * numCoords, SEEK_CUR);
#else
                fseek(in_, 39 * numCoords, SEEK_CUR);
#endif
            }
        }
    }
    // scan for element type

    while (!feof(in_))
    {

        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string tmp(buf);

        // scan for part token
        id = tmp.find("part");
        // part found - rewind one line and exit
        if (id != string::npos)
        {
            if (partList_ != NULL)
                partList_->push_back(actPart);
#ifdef WIN32
            _fseeki64(in_, -5L, SEEK_CUR);
#else
            fseek(in_, -5L, SEEK_CUR);
#endif
            return 0;
            ;
        }

        string elementType(strip(tmp));
        EnElement elem(elementType);

        // we have a valid ENSIGHT element
        if (elem.valid())
        {
            // get number of elements
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            int numElements = atoi(buf);
            int nc(elem.getNumberOfCorners());
            switch (elementId_)
            {
            case GIVEN:
#ifdef WIN32
                _fseeki64(in_, ((nc * 10) + 12) * numElements, SEEK_CUR);
#else
                fseek(in_, ((nc * 10) + 12) * numElements, SEEK_CUR);
#endif
                break;
            default:
#ifdef WIN32
                _fseeki64(in_, ((nc * 10) + 1) * numElements, SEEK_CUR);
#else
                fseek(in_, ((nc * 10) + 1) * numElements, SEEK_CUR);
#endif
            }
            actPart.addElement(elem, numElements);
        }
    } // if( elem.valid() )

    if (partList_ != NULL)
        partList_->push_back(actPart);

    return 0;
}
