/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class       EnGeoFile                 ++
// ++                                           En6GeoASC                 ++
// ++                                                                     ++
// ++ Author of initial version:  Ralf Mikulla (rm@vircinity.com)         ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 05.06.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include "GeoFileAsc.h"
#include "EnElement.h"

#include <util/coviseCompat.h>
#include <api/coModule.h>

using namespace std;

//////////////////////// geo file class ////////////////////////////////////

//
// Constructor
//
En6GeoASC::En6GeoASC(const coModule *mod)
    : EnFile(mod)
    , lineCnt_(0)
    , numCoords_(0)
    , indexMap_(NULL)
    , maxIndex_(0)
    , lastNc_(0)
{
    className_ = string("En6GeoASC");
}

En6GeoASC::En6GeoASC(const coModule *mod, const string &name)
    : EnFile(mod, name)
    , lineCnt_(0)
    , numCoords_(0)
    , indexMap_(NULL)
    , maxIndex_(0)
    , lastNc_(0)
{
    className_ = string("En6GeoASC");
}

//
// Destructor
//
En6GeoASC::~En6GeoASC()
{
    delete[] indexMap_;
}

void
En6GeoASC::read()
{
    // read header
    readHeader();
    // read coordinates
    En6GeoASC::readCoords();
    // read connectivity
    readConn();
    // TBD: err handling
}

// read header
int
En6GeoASC::readHeader()
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
En6GeoASC::readCoords()
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

//
// create a list of parts e.g.
//  - print a list of parts
//  - fill parts_
//
void
En6GeoASC::parseForParts()
{
    char buf[lineLen];
    int numParts(0);
    int totNumElements;

    EnPart *actPart(NULL);

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
            // length of "part" +1
            string pnumStr(strip(tmp.substr(id + 5)));
            actPartNr = atoi(pnumStr.c_str());

            // comment line we need it for the table output
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            string comment(buf);

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

            // add element info to the part
            if (actPart != NULL)
                actPart->addElement(elem, numElements);

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

            totNumElements += numElements;
            int i;
            // proceed number of elements lines
            for (i = 0; i < numElements; ++i)
            {
                fgets(buf, lineLen, in_);
                ++lineCnt_;
            }
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
}

// read connectivites
int
En6GeoASC::readConn()
{
    int ret(0);

    if (!isOpen_)
        return -1;

    // read until EOF is found
    char buf[lineLen];
    size_t id;

    int nEle(0);
    int *elePtr(NULL), *typePtr(NULL), *connPtr(NULL);
    int *elePtr2d(NULL), *typePtr2d(NULL), *connPtr2d(NULL);
    int *elePtr3d(NULL), *typePtr3d(NULL), *connPtr3d(NULL);
    int currEleIdx2d = 0, currEleIdx3d = 0, currConnIdx(0);

    int tNc, begNc(0);

    int *locArr = new int[21]; // an ENSIGHT element has max. 20 corners + 1 index
    int cornIn[20];
    int cornOut[20];

    int numElements;
    int nc;
    int covType;
    int onEle;
    int onCorner;
    int cnt(0);

    EnPart *actPart(NULL);

    // use the master part info to allocate element and corner lists
    if (masterPL_.empty())
        cerr << className_ << "::readConnX() MPL EMPTY!!! " << endl;

    bool partActive = false;

    vector<int> eleLst2d, eleLst3d, cornLst2d, cornLst3d, typeLst2d, typeLst3d;

    while (!feof(in_))
    {
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string tmp(buf);
        int actPartNr;

        // scan for part token
        id = tmp.find("part");
        if (id != string::npos)
        {
            // part line found
            // length of "part" +1
            string pnumStr(strip(tmp.substr(id + 5)));
            //cerr << className_ << "::readConn() found part NR: <" << pnumStr << ">" << endl;
            actPartNr = atoi(pnumStr.c_str());
            // get comment line
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            string comment(buf);
            strip(comment);
            id = string::npos;

            // add part to the partList
            // this happens after the part is completely constructed
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

                    eleLst2d.clear();
                    eleLst3d.clear();
                    typeLst2d.clear();
                    typeLst3d.clear();
                    cornLst2d.clear();
                    cornLst3d.clear();

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
                    // set active flag
                    EnPart &theMasterPart(masterPL_[ii]);
                    partActive = theMasterPart.isActive();
                    actPart->activate(partActive);
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
        // we have a valid ENSIGHT element
        if (elem.valid())
        {
            // get number of elements
            fgets(buf, lineLen, in_);
            ++lineCnt_;

            numElements = atoi(buf);

            if (actPart != NULL)
                actPart->addElement(elem, numElements);

            nc = elem.getNumberOfCorners();
            covType = elem.getCovType();
            // read the connectivity
            int i;
            for (i = 0; i < numElements; ++i)
            {
                fgets(buf, lineLen, in_);
                ++lineCnt_;
                // an integer always has 8 figures (see ENSIGHT docu)
                switch (elementId_)
                {
                case ASSIGN:
                    tNc = nc;
                    begNc = 0;
                    atoiArr(8, buf, locArr, nc);
                    break;
                case GIVEN:
                    // in this case locArr[0] contains the element id
                    tNc = nc + 1;
                    begNc = 1;
                    atoiArr(8, buf, locArr, tNc);
                    break;
                case OFF:
                    // !!!!! find out what to do here
                    atoiArr(8, buf, locArr, nc);
                    break;
                case EN_IGNORE:
                    // !!!! check this !!
                    atoiArr(8, buf, locArr, nc);
                    break;
                }

                // remap indicees (Ensight elements may have a different numbering scheme
                //                 as COVISE elements)
                int k(0);
                //  prepare arrays
                int j;
                for (j = begNc; j < tNc; ++j)
                {
                    switch (nodeId_)
                    {
                    case ASSIGN:
                        cornIn[k] = locArr[j] - 1; // we read FORTRAN output!!!

                        break;
                    case GIVEN:
                        cornIn[k] = indexMap_[locArr[j]];
                        break;
                    case OFF:
                        cornIn[k] = locArr[j] - 1; // we read FORTRAN output!!!  TRUE??
                        break;
                    case EN_IGNORE:
                        cornIn[k] = locArr[j] - 1; // we read FORTRAN output!!!  TRUE??
                        break;
                    }
                    k++;
                }
                // do the remapping
                elem.remap(cornIn, cornOut);

                if (partActive)
                {
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
        }
        // the first part will contain ALL coordinates
        // check if this concept is useful
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
    delete[] locArr;

    return ret;
}

// helper converts char buf containing num ints of length int_leng to int-array arr
void
En6GeoASC::atoiArr(const int &int_leng, char *buf, int *arr, const int &num)
{
    if ((buf != NULL) && (arr != NULL))
    {
        int cnt = 0, i = 0;
        string str(buf);
        //cerr << "En6GeoASC::atoiArr(..) STR: " << str << endl;
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
            cerr << "En6GeoASC::atoiArr(..) WARNING: found more numberes than expected!" << endl;
        // chunk will not be empty
        if (!chunk.empty())
            arr[i] = atoi(chunk.c_str());
        //cerr << "chunk: " << chunk << " int: " << i << " val: " << arr[i] << endl;
    }
}

int
En6GeoASC::readPart()
{
    int ret(0);
    if (isOpen_)
    {
        char buf[lineLen];
        int noRead(0);
        // 2 lines decription - ignore it
        fgets(buf, lineLen, in_);
        ++lineCnt_;
        string line(buf);
        if (line.find("part") != string::npos)
        {
            // part No
            fgets(buf, lineLen, in_);
            ++lineCnt_;
            int partNo;
            noRead = sscanf(buf, "%d", &partNo);
            cerr << className_ << "::readPart() got part No: " << partNo << endl;
            if (noRead != 1)
            {
                cerr << className_ << "::readPart() Error reading part No" << endl;
                return -1;
            }
            // description line
            fgets(buf, lineLen, in_);
            ++lineCnt_;
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
            cerr << className_ << "::readPart() got No. of coordinates (per part): " << nc << endl;
            if (noRead != 1)
            {
                cerr << className_ << "::readPart() Error reading no of coordinates" << endl;
                return -1;
            }

            int onc(numCoords_);
            numCoords_ += nc;

            // allocate memory
            float *tmp = new float[numCoords_];
            if (dc_.x != NULL)
            {
                memcpy(tmp, dc_.x, onc * sizeof(float));
                delete[] dc_.x;
            }
            dc_.x = tmp;

            tmp = new float[numCoords_];
            if (dc_.y != NULL)
            {
                memcpy(tmp, dc_.y, onc * sizeof(float));
                delete[] dc_.y;
            }
            dc_.y = tmp;

            tmp = new float[numCoords_];
            if (dc_.z != NULL)
            {
                memcpy(tmp, dc_.z, onc * sizeof(float));
                delete[] dc_.z;
            }
            dc_.z = tmp;
            dc_.setNumCoord(numCoords_);

            // we don't allocate indexMap_
            // fillIndexMap_ will do everything for us

            // id's or coordinates
            float val;
            int i;
            switch (nodeId_)
            {
            case ASSIGN:
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
                    dc_.x[i + onc] = val;
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
                    dc_.y[i + onc] = val;
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
                    dc_.z[i + onc] = val;
                }
                break;
            case GIVEN:
                // index array
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
                    fillIndexMap(iVal, i);
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
                    dc_.x[i + onc] = val;
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
                    dc_.y[i + onc] = val;
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
                    dc_.z[i + onc] = val;
                }
                break;
            case OFF:
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
                    dc_.x[i + onc] = val;
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
                    dc_.y[i + onc] = val;
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
                    dc_.z[i + onc] = val;
                }

                break;
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
                    dc_.x[i + onc] = val;
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
                    dc_.y[i + onc] = val;
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
                    dc_.z[i + onc] = val;
                }

                break;
            }
        }

        else
        {
            cerr << className_ << "::readPart() " << lineCnt_ << " NO part header found" << endl;
        }
        cerr << className_ << "::readPart()  got " << numCoords_ << " coordinates" << endl;
    }

    readPartConn();

    return ret;
}

int
En6GeoASC::readPartConn(void)
{
    return 0;
}

void
En6GeoASC::fillIndexMap(const int &i, const int &natIdx)
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
        cerr << " Test (En6GeoASC) number of arguments wrong: expect only filename" << endl;
        exit(-1);
    }

    string filen(argv[1]);

    cerr << " Test (En6GeoASC) try to open file <" << filen << "> " << endl;
    En6GeoASC enf(filen);

    enf.read();
}
#endif
