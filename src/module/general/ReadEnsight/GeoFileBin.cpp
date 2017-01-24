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

#include "GeoFileBin.h"
#include "GeoFileAsc.h"
#include <algorithm>
#include <functional>
#include <string>
using namespace std;

#if defined(__GNUC__) && (__GNUC__ > 4 || __GNUC__ == 4 && __GNUC_MINOR__ > 2)
#pragma GCC diagnostic warning "-Wuninitialized"
#endif

//
// Constructor
//
En6GeoBIN::En6GeoBIN(const coModule *mod, const string &name, EnFile::BinType binType)
    : EnFile(mod, binType)
    , indexMap_(NULL)
    , resetAllocInc_(true)
    , allocOffset_(10000)
    , debug_(false)
{
    className_ = string("En6GeoBIN");
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
        cerr << className_ << "::En6GeoBIN(..) open NOT successful" << endl;
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
       cerr << className_ << "::En6GeoBIN(..) unknown byte-order" << endl;
   }*/

    debug_ = (getenv("READ_ENSIGHT_DEBUG") != NULL);
}

//
// create a list of parts e.g.
//  - print a list of parts
//  - fill parts_
//
void
En6GeoBIN::parseForParts()
{
    if (!isOpen_)
        return;

    int numParts(0);
    int totNumElements;
    size_t id;

    EnPart *actPart(NULL);

    if (readHeader() < 0)
    {
        cerr << className_
             << "::parseForParts() WARNING header SUSPICIOUS!!!!"
             << " file may be corrupt or byteswap needed" << endl;
        fileMayBeCorrupt_ = true;
        return;
    }

    readCoordsDummy();
    int i, j, k;
    int d = 0;
    bool validElementFound = false;
    while (!feof(in_))
    {
        string tmp;
        try
        {
            tmp = getStr();
        }
        catch (InvalidWordException e)
        {
        }
        int actPartNr;
        cerr << className_ << "::parseForParts() - tmp Str - " << tmp << endl;

        // scan for part token
        id = tmp.find("part");
        if (id != string::npos)
        {

            string pnumStr(strip(tmp.substr(id + 5)));
            actPartNr = atoi(pnumStr.c_str());

            // comment line we need it for the table output
            string comment(getStr().substr(0, 80));

            if (partList_ != NULL)
            {
                if (actPart != NULL)
                {
                    partList_->push_back(*actPart);
                    delete actPart;
                }
            }
            actPart = new EnPart(actPartNr, comment);

            // now increment number of parts and reset element counter
            numParts++;
            totNumElements = 0;
        }
        else
        {
            d++;
        }

        string elementType(strip(tmp));
        EnElement elem(elementType);

        // we have a valid ENSIGHT element
        if (elem.valid())
        {
            validElementFound = true;
            // get number of elements
            int numElements = getInt();
            // add element info to the part
            if (actPart != NULL)
                actPart->addElement(elem, numElements);
            int nc(elem.getNumberOfCorners());
            totNumElements += numElements;
            // we set the number of 2/3D elements to 1 if
            // we find the cooresponding element type that's sufficient to mark
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
            // proceed number of elements lines
            switch (elementId_)
            {
            case ASSIGN:
                skipInt(numElements * nc);
                break;
            case GIVEN:
                skipInt(numElements);
                skipInt(numElements * nc);
                break;
            case OFF:
                // !!!!! find out what to do here
                skipInt(numElements * nc);
                break;
            case EN_IGNORE:
                // !!!! check this !!
                skipInt(numElements * nc);
                break;
            }
        } // if( elem.valid() )
        else if ((elementType.find("block")) < elementType.size())
        {
            cerr << "parseForParts::elementType is " << elementType << endl;
            int num = 3;
            int ijk[3];
            getIntArr(num, ijk);
            i = ijk[0];
            j = ijk[1];
            k = ijk[2];
            cerr << "i= " << i << endl;
            cerr << "j= " << j << endl;
            cerr << "k= " << k << endl;
            //numCoord_=i*j*k;
            skipFloat(i * j * k);
            skipFloat(i * j * k);
            skipFloat(i * j * k);
        }
    }

// rewind
#ifdef WIN32
    _fseeki64(in_, 0L, SEEK_SET);
#else
    fseek(in_, 0L, SEEK_SET);
#endif

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

// read header
int
En6GeoBIN::readHeader()
{
    int ret(0);
    if (isOpen_)
    {
        string line;
        try
        {
            // binary type
            line = getStr();
            std::transform(line.begin(), line.end(), line.begin(), (int (*)(int))tolower);
            if (line.find("binary") == string::npos)
            {
                return -1;
            }
            // 2 lines decription - ignore it
            line = getStr();
            line = getStr();
        }
        catch (InvalidWordException e)
        {
            cerr << className_ + "::readHeader() " + e.what() << endl;
            return -1;
        }

        // node id

        string tmp(getStr());
        if (tmp.size() == 0)
            return -1;
        char tok[20];
        strcpy(tok, " id");
        size_t beg(tmp.find(tok));

        if (beg == string::npos)
        {
#ifdef WIN32
            DebugBreak();
#endif
            //	    cerr << className_ << "::readHeader() ERROR node-id not found" << endl;
            return ret;
        }
        // " id"  has length 3
        beg += 4;
        string cmpStr(strip(tmp.substr(beg)));
        nodeId_ = -1; ////////
        if (cmpStr.find("off") != string::npos)
        {
            nodeId_ = EnFile::OFF;
            //	    cerr << "En6GeoBIN::readHeader() NODE ID OFF" << endl;
        }
        else if (cmpStr.find("given") != string::npos)
        {
            nodeId_ = EnFile::GIVEN;
            //	    cerr << "En6GeoBIN::readHeader() NODE ID GIVEN" << endl;
        }
        else if (cmpStr.find("assign") != string::npos)
        {
            nodeId_ = EnFile::ASSIGN;
            //	    cerr << "En6GeoBIN::readHeader() NODE ID ASSIGN" << endl;
        }
        else if (cmpStr.find("ignore") != string::npos)
        {
            nodeId_ = EnFile::EN_IGNORE;
            //	    cerr << "En6GeoBIN::readHeader() NODE ID IGNORE" << endl;
        }

        // element id
        tmp = getStr();

        beg = tmp.find(tok);

        if (beg == string::npos)
        {
#ifdef WIN32
            DebugBreak();
#endif
            cerr << className_ << "::readHeader() ERROR element-id not found" << endl;
            return ret;
        }

        beg += 4;
        cmpStr = strip(tmp.substr(beg));
        if (cmpStr.find("off") != string::npos)
        {
            elementId_ = EnFile::OFF;
            //	    cerr << "En6GeoBIN::readHeader() ELEMENT ID OFF" << endl;
        }
        else if (cmpStr.find("given") != string::npos)
        {
            elementId_ = EnFile::GIVEN;
            //	    cerr << "En6GeoBIN::readHeader() ELEMENT ID GIVEN" << endl;
        }
        else if (cmpStr.find("assign") != string::npos)
        {
            elementId_ = EnFile::ASSIGN;
            //	    cerr << "En6GeoBIN::readHeader() ELEMENT ID ASSIGN" << endl;
        }
        else if (cmpStr.find("ignore") != string::npos)
        {
            elementId_ = EnFile::EN_IGNORE;
            //	    cerr << "En6GeoBIN::readHeader() ELEMENT ID IGNORE" << endl;
        }
        else
        {
#ifdef WIN32
            DebugBreak();
#endif
            cerr << className_ << "::readHeader()  ELEMENT-ID Error" << endl;
        }

        // coordinates token
        tmp = getStr();
        strcpy(tok, "coordinates");
        beg = tmp.find_first_of(tok);

        if (beg == string::npos)
        {
#ifdef WIN32
            DebugBreak();
#endif
            cerr << className_ << "::readHeader() ERROR coordinates token not found" << endl;
            return ret;
        }

        // number of coordinates
        numCoords_ = getInt();
        cerr << className_ << "::readHeader()  got " << numCoords_ << " coordinates" << endl;
        ret = 1;
    }

    return ret;
}

// read coordinates
int
En6GeoBIN::readCoords()
{
    int ret(0);
    // we read only if we have a valid file and coordinates
    if ((isOpen_) && (numCoords_ > 0))
    {

        // allocate arrays
        // cleanup will be made by DataCont::cleanAll()
        dc_.setNumCoord(numCoords_);
        try
        {
            dc_.x = new float[numCoords_];
            dc_.y = new float[numCoords_];
            dc_.z = new float[numCoords_];
        }
        catch (std::exception &e)
        {
            cerr << "Exception (alloc of coordinate arrays Line: __LINE__) : " << e.what();
        }
        int *tmpIdxMap(NULL);

        float *coords(NULL);
        switch (nodeId_)
        {
		case ASSIGN:
		case OFF:
            coords = new float[3 * numCoords_];
            getFloatArr(3 * numCoords_, coords);
            break;
        case GIVEN:
            tmpIdxMap = new int[numCoords_];
            getIntArr(numCoords_, tmpIdxMap);
            coords = new float[3 * numCoords_];
            getFloatArr(3 * numCoords_, coords);
            break;
        default:
            return -1;
            break;
        }

        int cnt(0);
        for (int i = 0; i < 3 * numCoords_; i += 3)
        {
            dc_.x[cnt] = coords[i];
            dc_.y[cnt] = coords[i + 1];
            dc_.z[cnt] = coords[i + 2];
            cnt++;
        }
        // fill the index map now
        int idx(0);
        if (tmpIdxMap != NULL)
        {
            // pre allocate indexMap_ to a reasonable size
            // this avoids that the indexMap_ has to be reallocated several times
            // which is extremely time consuming
            int idxMax = *(std::max_element(tmpIdxMap, tmpIdxMap + numCoords_));
            maxIndex_ = idxMax + 10; // we alloc 10 ints more 'cause fillIdxMap checks >= and are surely on the safe side
            try
            {
                indexMap_ = new int[maxIndex_];
            }
            catch (std::exception &e)
            {
                cerr << "Exception (alloc of indexMap Line: __LINE__) : " << e.what();
                maxIndex_ = 0;
            }
            if (debug_)
                cerr << "En6GeoBIN::readCoords(): allocated indexMap_ to "
                     << maxIndex_ << " elements (" << maxIndex_ * sizeof(int) << "  bytes)" << endl;

            for (int i = 0; i < numCoords_; ++i)
            {
                idx = tmpIdxMap[i];
                fillIndexMap(idx, i);
            }
        }

        // delete temporary arrays
        delete[] coords;
        delete[] tmpIdxMap;
    }

    cerr << className_ << "::readCoords() got " << dc_.getNumCoord() << " coordinates" << endl;
    return ret;
}

// read coordinates
int
En6GeoBIN::readCoordsDummy()
{
    int ret(0);
    // we read only if we have a valid file and coordinates
    if ((isOpen_) && (numCoords_ > 0))
    {
        switch (nodeId_)
        {
        case ASSIGN:
		case OFF:
            skipFloat(3 * numCoords_);
            break;
        case GIVEN:
            skipInt(numCoords_);
            skipFloat(3 * numCoords_);
            break;
        default:
            return -1;
            break;
        }
    }
    //cerr << className_ << "::readCoordsDummy() done " << endl;
    return ret;
}

// read connectivities into parts
int
En6GeoBIN::readConn()
{
    int ret(0);

    if (!isOpen_)
        return -1;

    // read until EOF is found
    size_t id;

    int nEle(0);
    int *elePtr(NULL), *typePtr(NULL), *connPtr(NULL);
    int *elePtr2d(NULL), *typePtr2d(NULL), *connPtr2d(NULL);
    int *elePtr3d(NULL), *typePtr3d(NULL), *connPtr3d(NULL);
    int currEleIdx2d = 0, currEleIdx3d = 0, currConnIdx(0);

    int *locArr(NULL);
    int cornIn[20];
    int cornOut[20];

    int numElements;
    int nc;
    int covType;
    int onEle;
    int onCorner;
    int cnt(0);

    EnPart *actPart(NULL);
    if (masterPL_.empty())
        cerr << className_ << "::readConn() masterPL_ is empty() " << endl;

    bool partActive(false);

    vector<int> eleLst2d, eleLst3d, cornLst2d, cornLst3d, typeLst2d, typeLst3d;

    while (!feof(in_))
    {
        string tmp;
        try
        {
            tmp = getStr();
        }
        catch (InvalidWordException e)
        {
        }

        int actPartNr;

        // scan for part token
        id = tmp.find("part");
        if (id != string::npos)
        {
            // part line found
            // length of "part" +1
            string pnumStr(strip(tmp.substr(id + 5)));
            cerr << className_ << "::readConn() found part NR: <" << pnumStr.substr(0, 8) << ">" << endl;
            actPartNr = atoi(pnumStr.c_str());
            // get comment line
            string comment(getStr());
            if (comment.size() > 80)
                comment = comment.substr(0, 79);
            strip(comment);

            id = string::npos;
            // add part to the partList
            if (partList_ != NULL)
            {
                if (actPart != NULL)
                {
                    // create arrys explicitly
                    elePtr2d = new int[eleLst2d.size()];
                    elePtr3d = new int[eleLst3d.size()];
                    typePtr2d = new int[typeLst2d.size()];
                    typePtr3d = new int[typeLst3d.size()];
                    connPtr2d = new int[cornLst2d.size()];
                    connPtr3d = new int[cornLst3d.size()];

                    copy(eleLst2d.begin(), eleLst2d.end(), elePtr2d);
                    copy(eleLst3d.begin(), eleLst3d.end(), elePtr3d);
                    copy(typeLst2d.begin(), typeLst2d.end(), typePtr2d);
                    copy(typeLst3d.begin(), typeLst3d.end(), typePtr3d);
                    copy(cornLst2d.begin(), cornLst2d.end(), connPtr2d);
                    copy(cornLst3d.begin(), cornLst3d.end(), connPtr3d);
                    actPart->setNumEleRead2d(eleLst2d.size());
                    actPart->setNumEleRead3d(eleLst3d.size());
                    actPart->setNumConnRead2d(cornLst2d.size());
                    actPart->setNumConnRead3d(cornLst3d.size());
                    actPart->el2d_ = elePtr2d;
                    actPart->tl2d_ = typePtr2d;
                    actPart->cl2d_ = connPtr2d;
                    actPart->el3d_ = elePtr3d;
                    actPart->tl3d_ = typePtr3d;
                    actPart->cl3d_ = connPtr3d;

                    partList_->push_back(*actPart);
                    delete actPart;
                    actPart = NULL;

                    currEleIdx2d = 0;
                    currEleIdx3d = 0;
                    currConnIdx = 0;
                    onEle = 0;
                    onCorner = 0;
                    nEle = 0;
                    elePtr = NULL;
                    connPtr = NULL;
                    typePtr = NULL;
                }
            }
            actPart = new EnPart(actPartNr, comment);

            // find part with actPartNr in masterPL_
            unsigned int ii;
            for (ii = 0; ii < masterPL_.size(); ++ii)
            {
                if (masterPL_[ii].getPartNum() == actPartNr)
                {
                    EnPart &theMasterPart(masterPL_[ii]);
                    partActive = masterPL_[ii].isActive();
                    actPart->activate(partActive);
                    //if ( partActive )
                    //    cerr << className_ << "::readConnX() will read active part " << actPartNr << endl;
                    // set array sizes for elemen and cornerlists from MasterPartList
                    eleLst2d.resize(theMasterPart.getTotNumEle2d());
                    eleLst3d.resize(theMasterPart.getTotNumEle3d());
                    cornLst2d.resize(theMasterPart.getTotNumCorners2d());
                    cornLst3d.resize(theMasterPart.getTotNumCorners3d());
                    typeLst2d.resize(theMasterPart.getTotNumEle2d());
                    typeLst3d.resize(theMasterPart.getTotNumEle3d());
                }
            }
            cnt++;
        }

        // scan for element type
        string elementType(strip(tmp));
        EnElement elem(elementType);
        // we have a valid ENSIGHT element (type)
        if (elem.valid())
        {
            // get number of elements
            numElements = getInt();
            //cerr << className_ << "::readConn()  " << elem.getEnTypeStr() << "  "  <<  numElements << endl;

            if (actPart != NULL)
                actPart->addElement(elem, numElements);

            nc = elem.getNumberOfCorners();
            covType = elem.getCovType();
            // read the connectivity
            int arrSize(numElements * nc);
            if (locArr != NULL)
                cerr << "En6GeoBIN::readConnX(..) WARNING locArr != NULL before allocation" << endl;

            try
            {
                locArr = new int[arrSize];
            }
            catch (std::exception &e)
            {
                cout << "Exception (alloc of locArr Line: __LINE__) : " << e.what();
            }

            if (locArr == NULL)
            {
                cerr << "En6GeoBIN::readConn(..) cannot allocate locArr" << endl;
            }
            //cerr << "En6GeoBIN::readConn(..) allocated locarr " << arrSize << endl;
            if (partActive)
            {
                switch (elementId_)
                {
                case ASSIGN:
                    getIntArr(arrSize, locArr);
                    break;
                case GIVEN:
                    //getIntArr( numElements, NULL);
                    skipInt(numElements);
                    getIntArr(arrSize, locArr);
                    break;
                case OFF:
                    // !!!!! find out what to do here
                    getIntArr(arrSize, locArr);
                    break;
                case EN_IGNORE:
                    // !!!! check this !!
                    getIntArr(arrSize, locArr);
                    break;
                }
            }
            else
            {
                // proceed number of elements lines
                switch (elementId_)
                {
                case ASSIGN:
                    skipInt(numElements * nc);
                    break;
                case GIVEN:
                    skipInt(numElements);
                    skipInt(numElements * nc);
                    break;
                case OFF:
                    // !!!!! find out what to do here
                    skipInt(numElements * nc);
                    break;
                case EN_IGNORE:
                    // !!!! check this !!
                    skipInt(numElements * nc);
                    break;
                }
            }

            if (partActive)
            {
                // fill corner list
                int locCnt(0);
                int i, j;
                for (i = 0; i < numElements; ++i)
                {
                    switch (elementId_)
                    {
                    case ASSIGN:
                        for (j = 0; j < nc; ++j)
                        {
                            // we read a fortran array
                            cornIn[j] = locArr[locCnt] - 1;
                            locCnt++;
                        }
                        break;
                    case GIVEN:
                        for (j = 0; j < nc; ++j)
                        {
                            // we read a fortran array
                            cornIn[j] = indexMap_[locArr[locCnt]];
                            locCnt++;
                        }
                        break;
                    case OFF:
                        for (j = 0; j < nc; ++j)
                        {
                            // we read a fortran array
                            cornIn[j] = locArr[locCnt] - 1;
                            locCnt++;
                        }
                        break;
                    case EN_IGNORE:
                        for (j = 0; j < nc; ++j)
                        {
                            // we read a fortran array
                            cornIn[j] = locArr[locCnt] - 1;
                            locCnt++;
                        }
                        break;
                    }

                    elem.remap(cornIn, cornOut);
                    // assign corner list
                    if (elem.getDim() == EnElement::D2)
                    {
                        for (j = 0; j < nc; ++j)
                            cornLst2d.push_back(cornOut[j]);
                        eleLst2d.push_back(currEleIdx2d);
                        typeLst2d.push_back(covType);
                        currEleIdx2d += nc;
                    }
                    else if (elem.getDim() == EnElement::D3)
                    {
                        for (j = 0; j < nc; ++j)
                            cornLst3d.push_back(cornOut[j]);
                        eleLst3d.push_back(currEleIdx3d);
                        typeLst3d.push_back(covType);
                        currEleIdx3d += nc;
                    }
                }
            }
            delete[] locArr;
            locArr = NULL;
        }
        else if ((elementType.find("block")) < elementType.size())
        {
            cerr << "elementType is " << elementType << endl;
            int num = 3;
            int ijk[3];
            int i, j, k;
            getIntArr(num, ijk);
            i = ijk[0];
            j = ijk[1];
            k = ijk[2];
            cerr << "i= " << i << endl;
            cerr << "j= " << j << endl;
            cerr << "k= " << k << endl;
            //numCoord_=i*j*k;
            skipFloat(i * j * k);
            skipFloat(i * j * k);
            skipFloat(i * j * k);
        }
    }

    if (partList_ != NULL)
    {
        // add last part to the partList
        if ((actPart != NULL))
        {
            // create arrys explicitly
            elePtr2d = new int[eleLst2d.size()];
            elePtr3d = new int[eleLst3d.size()];
            typePtr2d = new int[typeLst2d.size()];
            typePtr3d = new int[typeLst3d.size()];
            connPtr2d = new int[cornLst2d.size()];
            connPtr3d = new int[cornLst3d.size()];

            copy(eleLst2d.begin(), eleLst2d.end(), elePtr2d);
            copy(eleLst3d.begin(), eleLst3d.end(), elePtr3d);
            copy(typeLst2d.begin(), typeLst2d.end(), typePtr2d);
            copy(typeLst3d.begin(), typeLst3d.end(), typePtr3d);
            copy(cornLst2d.begin(), cornLst2d.end(), connPtr2d);
            copy(cornLst3d.begin(), cornLst3d.end(), connPtr3d);
            actPart->setNumEleRead2d(eleLst2d.size());
            actPart->setNumEleRead3d(eleLst3d.size());
            actPart->setNumConnRead2d(cornLst2d.size());
            actPart->setNumConnRead3d(cornLst3d.size());
            actPart->el2d_ = elePtr2d;
            actPart->tl2d_ = typePtr2d;
            actPart->cl2d_ = connPtr2d;
            actPart->el3d_ = elePtr3d;
            actPart->tl3d_ = typePtr3d;
            actPart->cl3d_ = connPtr3d;

            partList_->push_back(*actPart);
            delete actPart;
            actPart = NULL;
            eleLst2d.clear();
            eleLst3d.clear();
            typeLst2d.clear();
            typeLst3d.clear();
            cornLst2d.clear();
            cornLst3d.clear();
        }
        // the first part will contain ALL coordinates
        // check if this concept is usefull
        if (!partList_->empty())
        {
            PartList::iterator beg = partList_->begin();
            if ((dc_.x != NULL) && (dc_.y != NULL) && (dc_.z != NULL))
            {
                beg->x3d_ = dc_.x;
                beg->y3d_ = dc_.y;
                beg->z3d_ = dc_.z;
                beg->setNumCoords(dc_.getNumCoord());
            }
        }
    }
    return ret;
}

void
En6GeoBIN::read()
{
    readHeader();

    readCoords();

    readConn();
}

//
//Destructor
//
En6GeoBIN::~En6GeoBIN()
{
    delete[] indexMap_;
}

void
En6GeoBIN::checkAllocOffset()
{
    char *nStr = getenv("READ_ENSIGHT_IDX_ALLOC_INC");
    if (nStr)
    {
        bool ok = true;
        for (int i = 0; i < strlen(nStr); ++i)
            ok = ok && isdigit(nStr[i]);
        if (ok)
        {
            allocOffset_ = atoi(nStr);
            if (debug_)
            {
                cerr << "ReadEnsight: setting allocation increment for index lookup to: "
                     << allocOffset_ << endl;
            }
        }
    }
}

void
En6GeoBIN::fillIndexMap(const int &i, const int &natIdx)
{
    int offSet = allocOffset_;
    // initial
    if (maxIndex_ == 0)
    {
        maxIndex_ = numCoords_;
        try
        {
            indexMap_ = new int[maxIndex_];
        }
        catch (std::exception &e)
        {
            cerr << "Exception (alloc of indexMap Line: __LINE__) : " << e.what();
        }
    }
    // realloc
    if (i >= maxIndex_)
    {
        if (debug_)
        {
            cerr << "En6GeoBIN::fillIndexMap(..) reallocation indexMap up to "
                 << (i + offSet) * sizeof(int) << " bytes" << endl;
        }
        int *tmp = NULL;
        try
        {
            tmp = new int[i + offSet];
        }
        catch (std::exception &e)
        {
            cerr << "Exception (realloc of indexMap Line: __LINE__) : " << e.what();
            throw(e);
        }
        int j;
        for (j = 0; j < maxIndex_; ++j)
            tmp[j] = indexMap_[j];
        maxIndex_ = i + offSet;
        delete[] indexMap_;
        indexMap_ = tmp;
    }
    indexMap_[i] = natIdx;
}
