/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class       EnFile                    ++
// ++                                           DataCont                  ++
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
#include "GeoFileBin.h"
#include "MGeoFileAsc.h"
#include "MGeoFileBin.h"
#include "EnGoldGeoASC.h"
#include "EnGoldGeoBIN.h"
#include "EnElement.h"
#include "ReadEnsight.h"
#include "MEnGoldMPGASC.h"


#include <util/coviseCompat.h>
#include <api/coModule.h>

using namespace covise;
InvalidWordException::InvalidWordException(const string &type)
    : type_(type)
{
}

string
InvalidWordException::what()
{
    return "Inavalid word found in reading " + type_;
}

//////////////////////////// base class ////////////////////////////

// static member to create an Ensight geometry file
// use the information given in the case file
EnFile *
EnFile::createGeometryFile(ReadEnsight *mod, const CaseFile &c, const string &filename)
{
    EnFile *enf = new EnFile(mod, filename);
    // file type
    EnFile::BinType binType(enf->binType());
    // Ensight version
    int version(c.getVersion());
    const char* ending = strrchr(filename.c_str(), '.');


    if (ending && strncasecmp(ending,".mpg",4) == 0)
    {
        // Ensight 6
        if (version == CaseFile::gold)
        {
            switch (binType)
            {
            case EnFile::NOBIN:
                // close file
                delete enf;
                enf = new MEnGoldMPGASC(mod, filename);
                break;
            case EnFile::UNKNOWN:
                delete enf;
                return NULL;
                break;
            default:
                std::cerr << "ERROR: EnFile bintype not implemented." ;
                return NULL;
                break;
            }
        }
    }
    else if (filename.length() >= 5 && strncasecmp(filename.c_str() + filename.length() - 5, ".mgeo", 5) == 0)
    {
        // Ensight 6
        if (version == CaseFile::v6)
        {
            switch (binType)
            {
            case EnFile::CBIN:
                delete enf;
                enf = new En6MGeoBIN(mod, filename);
                break;
            case EnFile::FBIN:
                delete enf;
                enf = new En6MGeoBIN(mod, filename, binType);
                break;
            case EnFile::NOBIN:
                // close file
                delete enf;
                enf = new En6MGeoASC(mod, filename);
                break;
            case EnFile::UNKNOWN:
                delete enf;
                return NULL;
                break;
            }
        }
    }
    else
    {

        // Ensight 6
        if (version == CaseFile::v6)
        {
            switch (binType)
            {
            case EnFile::CBIN:
                delete enf;
                enf = new En6GeoBIN(mod, filename);
                break;
            case EnFile::FBIN:
                delete enf;
                enf = new En6GeoBIN(mod, filename, binType);
                break;
            case EnFile::NOBIN:
                // close file
                delete enf;
                enf = new En6GeoASC(mod, filename);
                break;
            case EnFile::UNKNOWN:
                delete enf;
                return NULL;
                break;
            }
        }
        // Ensight GOLD
        else if (version == CaseFile::gold)
        {
            switch (binType)
            {
            case EnFile::CBIN:
                delete enf;
                enf = new EnGoldGeoBIN(mod, filename);
                break;
            case EnFile::FBIN:
                delete enf;
                enf = new EnGoldGeoBIN(mod, filename, binType);
                break;
            case EnFile::NOBIN:
                // close file
                delete enf;
                enf = new EnGoldGeoASC(mod, filename);
                break;
            case EnFile::UNKNOWN:
                delete enf;
                return NULL;
                break;
            }
        }
    }
    return enf;
}

EnFile::EnFile(ReadEnsight *mod, const BinType &binType)
    : fileMayBeCorrupt_(false)
    , className_(string("EnFile"))
    , isOpen_(false)
    , binType_(binType)
    , byteSwap_(false)
    , partList_(NULL)
    , dim_(1)
    , activeAlloc_(true)
    , dataByteSwap_(false)
    , ens(mod)
{
}

EnFile::EnFile(ReadEnsight *mod, const string &name, const int &dim, const BinType &binType)
    : fileMayBeCorrupt_(false)
    , className_(string("EnFile"))
    , isOpen_(false)
    , binType_(binType)
    , byteSwap_(false)
    , partList_(NULL)
    , dim_(dim)
    , activeAlloc_(true)
    , dataByteSwap_(false)
    , ens(mod)
    , name_(name)
{
    if (binType != FBIN && binType != CBIN)
    { // reopen as ASCII else leave it in binary mode
        in_ = fopen(name_.c_str(), "r");
    }
    else
    {

#ifdef WIN32
        in_ = fopen(name_.c_str(), "rb");
#else
        in_ = fopen(name_.c_str(), "r");
#endif
    }
    if (in_)
    {
        isOpen_ = true;
    }
    else
    {
        cerr << className_ << "::EnFile(..1) open NOT successful" << endl;
    }
}

EnFile::EnFile(ReadEnsight *mod, const string &name, const BinType &binType)
    : fileMayBeCorrupt_(false)
    , className_(string("EnFile"))
    , isOpen_(false)
    , binType_(binType)
    , byteSwap_(false)
    , partList_(NULL)
    , dim_(1)
    , activeAlloc_(true)
    , ens(mod)
    , name_(name)
{

    if (binType != FBIN && binType != CBIN)
    { // reopen as ASCII else leave it in binary mode
        in_ = fopen(name_.c_str(), "r");
    }
    else
    {

#ifdef WIN32
        in_ = fopen(name_.c_str(), "rb");
#else
        in_ = fopen(name_.c_str(), "r");
#endif
    }
    if (in_)
    {
        isOpen_ = true;
    }
    else
    {
        cerr << className_ << "::EnFile(..2) open NOT successful" << endl;
    }
}


void
EnFile::createGeoOutObj(dimType dim, coDistributedObject **outObjects2d, coDistributedObject **outObjects3d, const string &actObjNm2d, const string &actObjNm3d, int &timeStep)
{

    ens->globalParts_.push_back(*partList_);
    // create DO's
    coDistributedObject **oOut = ens->createGeoOutObj(actObjNm2d, actObjNm3d, timeStep);
    outObjects2d[timeStep] = NULL;
    outObjects3d[timeStep] = NULL;
    if (oOut)
    {
        if (oOut[0] != NULL)
            outObjects3d[timeStep] = oOut[0];
        if (oOut[1] != NULL)
            outObjects2d[timeStep] = oOut[1];
    }
    ++timeStep;
    outObjects2d[timeStep] = NULL;
    outObjects3d[timeStep] = NULL; 
}
void EnFile::createDataOutObj(dimType dim, coDistributedObject ** outObjects, const string & baseName, int & timeStep, int numTimeSteps, bool perVertex)
{
    // create DO's
    coDistributedObject **oOut = ens->createDataOutObj(dim, baseName, dc_, timeStep, numTimeSteps,perVertex);

    dc_.cleanAll();

    outObjects[timeStep] = NULL;
    if (oOut[0] != NULL)
        outObjects[timeStep] = oOut[0];

    ++timeStep;
    outObjects[timeStep] = NULL;
}
void
EnFile::setActiveAlloc(const bool &b)
{
    activeAlloc_ = b;
}

bool
EnFile::isOpen()
{
    return isOpen_;
}

EnFile::BinType
EnFile::binType()
{
    BinType ret(binType_);
    if (binType_ == EnFile::UNKNOWN)
    {
        if (isOpen_)
        {
            //	    cerr << "EnFile::binType() have to close file first " << endl;
            fclose(in_);
        }
#ifdef WIN32
        in_ = fopen(name_.c_str(), "rb");
#else
        in_ = fopen(name_.c_str(), "r");
#endif
        if (in_)
        {
            isOpen_ = true;
        }
        else
        {
            cerr << className_ << "::binType() open NOT successful" << endl;
            return CBIN;
        }
        char buf[81];
        buf[80] = '\0';
        char c;
        int j(0);
        for (int i = 0; i < 80; ++i)
        {
            c = fgetc(in_);
            if (c > 0 && static_cast<unsigned char>(c) < 255 && isprint(c))
            {
                buf[j] = c;
                j++;
            }
        }
        string firstLine(buf);
        std::transform(firstLine.begin(), firstLine.end(), firstLine.begin(), tolower);

        if ((firstLine.find("c binary") != string::npos) || (firstLine.find("cbinary") != string::npos))
        {
            ret = CBIN;
        }
        else if ((firstLine.find("fortran binary") != string::npos) || (firstLine.find("fortranbinary") != string::npos))
        {
            ret = FBIN;
        }
        else
            ret = NOBIN;

        if (ret != FBIN && ret != CBIN)
        { // reopen as ASCII else leave it in binary mode
            fclose(in_);

            if (isOpen_)
                in_ = fopen(name_.c_str(), "r");
        }

        binType_ = ret;
    }
    return ret;
}

DataCont
EnFile::getDataCont() const
{
    return dc_;
}

EnFile::~EnFile()
{
    if (isOpen_)
        fclose(in_);
}

// helper skip n floats or doubles
void
EnFile::skipFloat(const uint64_t &n)
{
    if (binType_ == EnFile::FBIN)
    { // check for block markers
        int ilen(getIntRaw());

#ifdef WIN32
        _fseeki64(in_, ilen, SEEK_CUR);
#else
        fseek(in_, ilen, SEEK_CUR);
#endif

        int olen(getIntRaw());
        if ((ilen != olen))
        {
#ifdef WIN32
            DebugBreak();
#endif
            cerr << "ERROR: EnFile::skipFloat(): wrong number of elements in block, expected" << n << " but blockstart: " << ilen << " blockend: " << olen << endl;
        }
    }

    // Read floats up to 4GB
    else
    {
#ifdef WIN32
        _fseeki64(in_, n * sizeof(float), SEEK_CUR);
#else
        fseek(in_, n * sizeof(float), SEEK_CUR);
#endif
    }
}

// helper skip n ints
void
EnFile::skipInt(const uint64_t&n)
{
    if (binType_ == EnFile::FBIN)
    { // check for block markers
        int ilen(getIntRaw());
#ifdef WIN32
        _fseeki64(in_, n * 4, SEEK_CUR);
#else
        fseek(in_, n * 4, SEEK_CUR);
#endif

        int olen(getIntRaw());
        if ((ilen != olen) || (ilen != n * 4))
        {
#ifdef WIN32
            DebugBreak();
#endif
            cerr << "ERROR: EnFile::skipFloat(): wrong number of elements in block, expected" << n << " but blockstart: " << ilen << " blockend: " << olen << endl;
        }
    }
    else
    {
#ifdef WIN32
        _fseeki64(in_, n * sizeof(int), SEEK_CUR);
#else
        fseek(in_, n * sizeof(int), SEEK_CUR);
#endif
    }
}

// helper to read binary strings
string
EnFile::getStr()
{
    const int strLen(80);
    char buf[81];
    buf[80] = '\0';
    int olen(0);
    string ret;

    if (binType_ == EnFile::FBIN)
    {
        int ilen(getIntRaw());
        if (feof(in_))
        {
            //end of file reached
            return ret;
        }
        //automatic byteorder detection
        if (ilen == 0)
        {
            olen = getIntRaw();
            cerr << "WARNING: EnFile::getStr(): empty string" << endl;
            if (olen != 0)
            {
#ifdef WIN32
                DebugBreak();
#endif
                cerr << "ERROR: EnFile::getStr(): not a fortran string " << endl;
            }
            return ret;
        }
        if (ilen != strLen)
        { // we have to do byteswaping here
            byteSwap_ = true;
            byteSwap(ilen);
        }
        if (ilen != strLen)
        {
#ifdef WIN32
            DebugBreak();
#endif
            cerr << "ERROR: EnFile::getStr(): not a fortran string of length 80" << endl;
            return ret;
        }
        fread(buf, strLen, 1, in_);
        if (feof(in_))
        {
#ifdef WIN32
            DebugBreak();
#endif
            cerr << "ERROR: EnFile::getStr(): end of file during read of a fortran string of length 80" << endl;
            return ret;
        }
        olen = getIntRaw();
        if (ilen != olen)
        {
#ifdef WIN32
            DebugBreak();
#endif
            cerr << "ERROR: EnFile::getStr(): not a fortran string of length 80" << endl;
            return ret;
        }
    }
    else
    {
        for (int i = 0; i < strLen; ++i)
        {
            buf[i] = fgetc(in_);
        }
    }

    buf[79] = '\0';
    ret = buf;
    return ret;
}

int
EnFile::getInt()
{
    int ret;
    if (binType_ == EnFile::FBIN)
    {
        getIntRaw();
        ret = getIntRaw();
        getIntRaw();
    }
    else
    {
        ret = getIntRaw();
    }
    return ret;
}

int
EnFile::getIntRaw()
{
    int ret = 0;
    fread(&ret, 4, 1, in_); // read a 4 byte integer
    if (byteSwap_)
    {
        byteSwap(ret);
    }
    return ret;
}

int *
EnFile::getIntArr(const uint64_t &n, int *iarr)
{
    if ((n == 0) || (iarr == NULL))
        return NULL;

    if (binType_ == EnFile::FBIN)
    {
        if (in_)
        {
            int ilen(getIntRaw());
            getIntArrHelper(n, iarr);
            int olen(getIntRaw());
            if ((ilen != olen) && (!feof(in_)))
            {
#ifdef WIN32
                DebugBreak();
#endif
                cerr << "EnFile::getIntArr(..) length mismatch (fortran) " << endl;
            }
        }
        else
        {
            cerr << "EnFile::getIntArr(..) stream not in good condition" << endl;
        }
    }
    else
    {
        getIntArrHelper(n, iarr);
    }

    return NULL;
}

void
EnFile::getIntArrHelper(const uint64_t &n, int *iarr)
{
    //////////////////////////// quick workaround (the original code doesn't work if n*sizeof(int) exceeds maxint)
    // TODO: we now use fread() -> is this still nescessary?
    if (n > 100000000)
    {
        uint64_t offset(0);
        while (offset < n)
        {
            uint64_t current = n - offset;
            if (current > 100000000)
                current = 100000000;
            getIntArrHelper(current, (iarr + offset));
            offset += current;
        }
        return;
    }
    ////////////////////////////

    const int len(sizeof(int));
    char *buf = new char[n * len];

    fread(buf, len, n, in_);

    if (byteSwap_)
        byteSwap((uint32_t *)buf, n);

    if (iarr != NULL)
        memcpy(iarr, buf, n * len);

    delete[] buf;
    return;
}

void
EnFile::setPartList(PartList *p)
{
    partList_ = p;
}

float *
EnFile::getFloatArr(const uint64_t &n, float *farr)
{
    if ((n == 0) || (farr == NULL))
        return NULL;

    const int len(sizeof(float));
    char *buf = new char[n * len];
    bool eightBytePerFloat = false;

    if (binType_ == EnFile::FBIN)
    {
        int ilen(getIntRaw());
        int olen;

        // we may have obtained double arrays
        // unfortionately ifstream.read will read only basic types
        // otherwise we would have saved one memcpy
        // There may be more elegant soutions for that.
        if (ilen / n == 8)
        {
            eightBytePerFloat = true;
            double *dummyArr = new double[n];
            delete[] buf;
            buf = new char[ilen];
            fread(buf, 8, n, in_);
            olen = getIntRaw();
            if (byteSwap_)
                byteSwap((uint64_t *)buf, n);
            memcpy(dummyArr, buf, ilen);
            cerr << "got 64-bit floats" << endl;
            int i;
            if (farr != NULL)
                for (i = 0; i < n; ++i)
                    farr[i] = (float)dummyArr[i];
            delete[] dummyArr;
        }
        else
        {
            fread(buf, len, n, in_);
            olen = getIntRaw();
        }
        if ((ilen != olen) && (!feof(in_)))
        {
#ifdef WIN32
            DebugBreak();
#endif
            cerr << "EnFile::getFloatArr(..) length mismatch (fortran) " << ilen << "     " << olen << endl;
        }
    }
    else
    {
        fread(buf, len, n, in_);
    }

    if (!eightBytePerFloat)
    {
        if (byteSwap_)
            byteSwap((uint32_t *)buf, n);
        if (farr != NULL)
            memcpy(farr, buf, n * len);
    }
    delete[] buf;
    return NULL;
}

// find a part by its part number
EnPart *
EnFile::findPart(const int &partNum) const
{
    if (partList_ != NULL)
    {
        unsigned int i;
        for (i = 0; i < partList_->size(); ++i)
        {
            if ((*partList_)[i].getPartNum() == partNum)
                return &(*partList_)[i];
        }
    }
    return NULL;
}

void
EnFile::resetPart(const int &partNum, EnPart *p)
{
    if (partList_ != NULL)
    {
        unsigned int i;
        for (i = 0; i < partList_->size(); ++i)
        {
            if ((*partList_)[i].getPartNum() == partNum)
            {
                (*partList_)[i] = *p;
                return;
            }
        }
    }
}

// find a part by its part number in the master part list
EnPart
EnFile::findMasterPart(const int &partNum) const
{
	unsigned int i;
	for (i = 0; i < ens->masterPL_.size(); ++i)
	{
		if ((ens->masterPL_)[i].getPartNum() == partNum)
			return ens->masterPL_[i];
	}
    EnPart dummy(-1);
    return dummy;
}

// allocate memory in the data container due to the collected part information
// USE this method ONLY for vertex based VARIABLE DATA
// fill data container (used only by ensight data classes)
// data values - unused parts will be filled by MAXFLT
// clean up
// TBD: find a better solution for part data!!! the used direct-access
//      pointer members are dangerous
void
EnFile::buildParts(const bool &isPerVert)
{
    if (partList_ != NULL)
    {
        uint64_t i;
        // allocate memory of the data container
        // 1st: find out how may values
        uint64_t totalNumberOfElements(0);
        for (i = 0; i < partList_->size(); ++i)
        {
            EnPart &p((*partList_)[i]);
            if (p.isActive())
            {
                uint64_t numEle;
                if (isPerVert)
                    numEle = p.numCoords();
                else
                    numEle = p.getTotNumEle();
                totalNumberOfElements += numEle;
            }
        }

        // 2nd: to be safe clean data container first
        if (dc_.x != NULL)
            delete[] dc_.x;
        if (dc_.y != NULL)
            delete[] dc_.y;
        if (dc_.z != NULL)
            delete[] dc_.z;

        // 3rd: do it
        //cerr << "EnFile::buildParts() will allocate dc with number of elements " << totalNumberOfElements << endl;
        switch (dim_)
        {
        case 1:
            dc_.setNumCoord(totalNumberOfElements);
            dc_.x = new float[totalNumberOfElements];
            break;
        case 3:
            dc_.setNumCoord(totalNumberOfElements);
            dc_.x = new float[totalNumberOfElements];
            dc_.y = new float[totalNumberOfElements];
            dc_.z = new float[totalNumberOfElements];
            break;
        }

        // now copy the part arrays to the data container
        int cnt(0);
        for (i = 0; i < partList_->size(); ++i)
        {
            EnPart &p((*partList_)[i]);
            if (p.isActive())
            {
                uint64_t numEle;
                if (isPerVert)
                    numEle = p.numCoords();
                else
                    numEle = p.getTotNumEle();

                int j;
                switch (dim_)
                {
                case 1:
                    if (p.arr1_ != NULL)
                    {
                        for (j = 0; j < numEle; ++j)
                        {
                            dc_.x[cnt] = p.arr1_[j];
                            cnt++;
                        }
                    }
                    else
                    {
                        for (j = 0; j < numEle; ++j)
                        {
                            dc_.x[cnt] = FLT_MAX;
                            cnt++;
                        }
                    }
                    // clean up
                    delete[] p.arr1_;
                    p.arr1_ = NULL;
                    break;
                case 3:
                    if (p.arr1_ != NULL)
                    {
                        for (j = 0; j < numEle; ++j)
                        {
                            dc_.x[cnt] = p.arr1_[j];
                            dc_.y[cnt] = p.arr2_[j];
                            dc_.z[cnt] = p.arr3_[j];
                            cnt++;
                        }
                    }
                    else
                    {
                        for (j = 0; j < numEle; ++j)
                        {
                            dc_.x[cnt] = FLT_MAX;
                            dc_.y[cnt] = FLT_MAX;
                            dc_.z[cnt] = FLT_MAX;
                            cnt++;
                        }
                    }
                    // clean up
                    delete[] p.arr1_;
                    p.arr1_ = NULL;
                    delete[] p.arr2_;
                    p.arr2_ = NULL;
                    delete[] p.arr3_;
                    p.arr3_ = NULL;
                    break;
                }
            }
            dc_.setNumCoord(cnt);
        }
    }
}

void
EnFile::setDataByteSwap(const bool & /*v*/)
{
    /* wird jetzt automatisch gemacht dataByteSwap_=v;
    if (dataByteSwap_) {
   byteSwap_ = ! byteSwap_;
    }*/
}

void
EnFile::setIncludePolyeder(const bool &b)
{
    includePolyeder_ = b;
}

void
EnFile::sendPartsToInfo()
{
    // output to covise covise info pannel
    char ostr[256];
    strcpy(ostr, "List of Ensight Parts:");
    ens->sendInfo("%s", ostr);

    strcpy(ostr, " ");
    ens->sendInfo("%s", ostr);

    strcpy(ostr, "  Ref# |  Part# | Part Description                                                                   | Number of Elements | Dimension");
    ens->sendInfo("%s", ostr);

    strcpy(ostr, "-------------------------------------------------------------------------------------------------------------------------------------------------------------------------");
    ens->sendInfo("%s", ostr);

    int cnt(0);
    if (partList_ != NULL)
    {
        vector<EnPart>::iterator pos(partList_->begin());
        for (; pos != partList_->end(); pos++)
        {
            ens->sendInfo("%s", (pos->partInfoString(cnt)).c_str());
            cnt++;
        }
    }
    ens->sendInfo("...Finished: List of Ensight Parts");
}

/////////////////////////// class DataCont /////////////////////////////////
DataCont::DataCont()
    : x(NULL)
    , y(NULL)
    , z(NULL)
    , el(NULL)
    , cl(NULL)
    , tl(NULL)
    , nCoord_(0)
    , nElem_(0)
    , nConn_(0)
{
}

DataCont::~DataCont()
{
    //// DONT CLEAN ANYTHING HERE !!!
}

void
DataCont::cleanAll()
{
    // clean up data container
    if (x)
        delete[] x;
    if (y)
        delete[] y;
    if (z)
        delete[] z;
    if (el)
        delete[] el;
    if (cl)
        delete[] cl;
    if (tl)
        delete[] tl;
}

// helper strip off spaces
string
strip(const string &str)
{
    string ret;
    if (!str.empty())
    {
        char *chtmp = new char[1 + str.size()];
        strcpy(chtmp, str.c_str());

        char *c = chtmp;
        char *pstart = chtmp;

        c += strlen(chtmp) - 1;

        while (c >= pstart)
        {
            if (*c > 0 && isspace(*c))
            {
                *c = '\0';
                c--;
            }
            else
                break;
        }

        while (*pstart > 0 && isspace(*pstart))
            pstart++;

        pstart[strlen(pstart)] = '\0';
        ret += pstart;

        delete[] chtmp;
    }
    return ret;
}
