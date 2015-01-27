/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                            ++
// ++             Implementation of class CaseFile                          ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date:                                                 ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "CaseFile.h"
#include <util/coviseCompat.h>

//
// helper
//
string trim(const string &s)
{
    int i;
    string tmp, ret;

    for (i = 0; i < s.size(); ++i)
    {
        if (!isspace(s[i]))
            break;
    }
    tmp = s.substr(i);

    for (i = tmp.size() - 1; i > 0; --i)
    {
        if (!isspace(tmp[i]))
            break;
    }
    ret = tmp.substr(0, i + 1);
    return ret;
}

//
// Constructor
//
CaseFile::CaseFile()
    : empty_(true)
    , geoTsIdx_(-1)
{
}

CaseFile::CaseFile(const CaseFile &cf)
    : empty_(cf.empty_)
    , geoFileNm_(cf.geoFileNm_)
    , mgeoFileNm_(cf.mgeoFileNm_)
    , dataIts_(cf.dataIts_)
    , version_(cf.version_)
    , fullFilename_(cf.fullFilename_)
    , dir_(cf.dir_)
    , timeSets_(cf.timeSets_)
    , geoTsIdx_(cf.geoTsIdx_)
{
}

void
CaseFile::setGeoFileNm(const string &fn)
{
    empty_ = false;
    geoFileNm_ = fn;
}

void
CaseFile::setMGeoFileNm(const string &fn)
{
    empty_ = false;
    mgeoFileNm_ = fn;
}

void
CaseFile::setVersion(const int &v)
{
    empty_ = false;
    version_ = v;
};

string
CaseFile::getGeoFileNm() const
{
    // return empty string if no filename is given
    if (geoFileNm_.empty())
    {
        string dummy;
        return dummy;
    }
    return dir_ + string("/") + geoFileNm_;
}

string
CaseFile::getMGeoFileNm() const
{
    // return empty string if no filename is given
    if (mgeoFileNm_.empty())
    {
        string dummy;
        return dummy;
    }
    return dir_ + string("/") + mgeoFileNm_;
}

//
// Method:
//
void
CaseFile::addDataIt(const DataItem &it)
{
    dataIts_.push_back(it);
}

string
CaseFile::printEnVersion()
{

    string out;
    switch (version_)
    {
    case CaseFile::v6:
        out = string("ensight v6");
        break;
    case CaseFile::gold:
        out = string("ensight gold");
        break;
    default:
        out = string("no version");
        break;
    }
    return out;
}

void
CaseFile::setFullFilename(const string &fn)
{
    fullFilename_ = trim(fn);
    size_t id(fullFilename_.find_last_of("/"));
    size_t id2(fullFilename_.find_last_of("\\"));
    if (id2 != string::npos)
    {
        if (id == string::npos || id2 > id)
            id = id2;
    }
    if (id == string::npos)
    {
        dir_ = string(".");
        projectNm_ = fullFilename_;
    }
    else
    {
        dir_ = fullFilename_.substr(0, id);
        projectNm_ = fullFilename_.substr(id);
        //	cerr << "CaseFile::setFullFilename(..) dir: " << dir_ << endl;
    }
    // remove trailing '/' or '\\' if there is one in dir_
    size_t dirlen(dir_.size());
    if (dir_[dirlen - 1] == '/' || dir_[dirlen - 1] == '\\')
        dir_ = dir_.substr(0, dirlen - 1);
}

string
CaseFile::getDir()
{
    return dir_;
}

string
CaseFile::getProjectName()
{
    return projectNm_;
}

void
CaseFile::setGeoTsIdx(const int &idx)
{
    geoTsIdx_ = idx;
}

int
CaseFile::getGeoTsIdx()
{
    return geoTsIdx_;
}

void
CaseFile::addTimeSet(TimeSet *ts)
{
    timeSets_.push_back(ts);
}

TimeSet *
CaseFile::getLastTimeSet()
{
    return timeSets_.back();
}

const TimeSets &
CaseFile::getAllTimeSets()
{
    return timeSets_;
}

//
// Destructor
//
CaseFile::~CaseFile()
{
}

/////////////////////////// class TimeSet //////////////////////////////

TimeSet::TimeSet(const int &n, const int &numTs)
    : num_(n)
    , numTs_(numTs)
{
}

TimeSet::TimeSet(const TimeSet &ts)
    : num_(ts.num_)
    , numTs_(ts.numTs_)
    , fileNums_(ts.fileNums_)
    , rTimes_(ts.rTimes_)
{
}

int
TimeSet::getIdx() const
{
    return num_;
}

void
TimeSet::addFileNr(const int &fn)
{
    fileNums_.push_back(fn);
}

void
TimeSet::addRealTimeVal(const float &rt)
{
    rTimes_.push_back(rt);
}

int
TimeSet::getNumTs() const
{
    return numTs_;
}

vector<string>
TimeSet::getFileNames(const string &t)
{
    // take care of ensight filename convention for time-dependent data
    vector<string> ret;

    size_t beg(t.find_first_of("*"));
    // return input name if NO asterix is found
    if (beg == string::npos)
    {
        ret.push_back(t);
        return ret;
    }

    size_t end(t.find_first_not_of("*", beg));
    if (end == string::npos)
    {
        end = t.size();
    }

    string pre(t.substr(0, beg));
    string stars(t.substr(beg, end - beg));
    string post(t.substr(end));

    int width(stars.size());

    vector<int>::iterator it;

    char ch[64]; // that's more then enough
    for (it = fileNums_.begin(); it != fileNums_.end(); it++)
    {
        sprintf(ch, "%0*d", width, *it);
        string fname(pre + string(ch) + post);
        ret.push_back(fname);
    }

    return ret;
}

vector<float>
TimeSet::getRealTimes() const
{
    return rTimes_;
}
