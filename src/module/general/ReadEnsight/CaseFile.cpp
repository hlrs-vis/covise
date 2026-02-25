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
#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <boost/algorithm/string.hpp>

//
// Constructor
//
CaseFile::CaseFile(): empty_(true), geoTsIdx_(-1)
{}

CaseFile::CaseFile(const CaseFile &cf)
: empty_(cf.empty_)
, geoFileNm_(cf.geoFileNm_)
, mgeoFileNm_(cf.mgeoFileNm_)
, dataIts_(cf.dataIts_)
, version_(cf.version_)
, fullFilename_(cf.fullFilename_)
, dir_(cf.dir_)
, projectNm_(cf.projectNm_)
, timeSets_(cf.timeSets_)
, fieldMap_(cf.fieldMap_)
, geoTsIdx_(cf.geoTsIdx_)
, binType_(cf.binType_)
, connectivityFileIndex_(cf.connectivityFileIndex_)
{}

void CaseFile::setGeoFileNm(const std::string &fn)
{
    empty_ = false;
    geoFileNm_ = fn;
}

void CaseFile::setMGeoFileNm(const std::string &fn)
{
    empty_ = false;
    mgeoFileNm_ = fn;
}

void CaseFile::setVersion(const int &v)
{
    empty_ = false;
    version_ = v;
}

void CaseFile::setConnectivityFileIndex(int idx)
{
    connectivityFileIndex_ = idx;
}

int CaseFile::getConnectivityFileIndex() const
{
    return connectivityFileIndex_;
}

std::string CaseFile::getGeoFileNm() const
{
    // return empty std::string if no filename is given
    if (geoFileNm_.empty()) {
        std::string dummy;
        return dummy;
    }
    return dir_ + std::string("/") + geoFileNm_;
}

std::string CaseFile::getMGeoFileNm() const
{
    // return empty std::string if no filename is given
    if (mgeoFileNm_.empty()) {
        std::string dummy;
        return dummy;
    }
    return dir_ + std::string("/") + mgeoFileNm_;
}

//
// Method:
//
void CaseFile::addDataItem(const DataItem &it)
{
    dataIts_[it.getDesc()] = it;
    fieldMap_[it.getDesc()] = it.getTimeSet();
    std::cerr << "CaseFile::addDataItem: " << it.getDesc() << ", time set: " << it.getTimeSet() << std::endl;
}

const DataItem *CaseFile::getDataItem(const std::string &field) const
{
    auto it = dataIts_.find(field);
    if (it == dataIts_.end())
        return nullptr;
    return &it->second;
}

std::string CaseFile::printEnVersion()
{
    std::string out;
    switch (version_) {
    case CaseFile::v6:
        out = std::string("ensight v6");
        break;
    case CaseFile::gold:
        out = std::string("ensight gold");
        break;
    default:
        out = std::string("no version");
        break;
    }
    return out;
}

void CaseFile::setFullFilename(const std::string &fn)
{
    fullFilename_ = boost::trim_copy(fn);
    size_t id(fullFilename_.find_last_of("/"));
    size_t id2(fullFilename_.find_last_of("\\"));
    if (id2 != std::string::npos) {
        if (id == std::string::npos || id2 > id)
            id = id2;
    }
    if (id == std::string::npos) {
        dir_ = std::string(".");
        projectNm_ = fullFilename_;
    } else {
        dir_ = fullFilename_.substr(0, id);
        projectNm_ = fullFilename_.substr(id);
        //	cerr << "CaseFile::setFullFilename(..) dir: " << dir_ << std::endl;
    }
    // remove trailing '/' or '\\' if there is one in dir_
    size_t dirlen(dir_.size());
    if (dir_[dirlen - 1] == '/' || dir_[dirlen - 1] == '\\')
        dir_ = dir_.substr(0, dirlen - 1);
}

std::string CaseFile::getDir() const
{
    return dir_;
}

std::string CaseFile::getProjectName()
{
    return projectNm_;
}

void CaseFile::setGeoTsIdx(const int &idx)
{
    geoTsIdx_ = idx;
}

int CaseFile::getGeoTsIdx() const
{
    return geoTsIdx_;
}

void CaseFile::addTimeSet(TimeSet *ts)
{
    timeSets_.push_back(ts);
    int idx = ts->getIdx();
    if (idx < 0) {
        std::cerr << "CaseFile::addTimeSet: invalid time set index " << idx << std::endl;
        idx = 0;
        return;
    }
    if (idx >= 0 && idx >= timeSets_.size()) {
        timeSets_.resize(idx + 1);
    }
    timeSets_[idx] = ts;
}

const TimeSet *CaseFile::getTimeSet(int idx) const
{
    return timeSets_[idx];
}

const TimeSet *CaseFile::getTimeSet(const std::string &field) const
{
    auto it = fieldMap_.find(field);
    if (it == fieldMap_.end()) {
        std::cerr << "CaseFile::getTimeSet: field " << field << ": NOT FOUND, #fields=" << fieldMap_.size()
                  << std::endl;
        return nullptr;
    }
    std::cerr << "CaseFile::getTimeSet: field " << field << ": mapping to idx " << it->second << std::endl;
    if (it->second < 0) {
        return nullptr;
    }
    if (it->second >= timeSets_.size()) {
        return nullptr;
    }
    return timeSets_[it->second];
}

const TimeSet *CaseFile::getGeoTimeSet() const
{
    if (geoTsIdx_ < 0)
        return nullptr;
    if (geoTsIdx_ >= timeSets_.size())
        return nullptr;
    return timeSets_[geoTsIdx_];
}

const TimeSet *CaseFile::getLastTimeSet() const
{
    return timeSets_.back();
}

TimeSet *CaseFile::getLastTimeSet()
{
    return timeSets_.back();
}

std::vector<float> CaseFile::getAllRealTimes() const
{
    std::set<float> times;
    for (auto ts = timeSets_.begin(); ts != timeSets_.end(); ts++) {
        std::vector<float> rts = (*ts)->getRealTimes();
        times.insert(rts.begin(), rts.end());
    }
    std::vector<float> ret;
    std::copy(times.begin(), times.end(), std::back_inserter(ret));
    return ret;
}

std::vector<std::string> CaseFile::makeFileNames(const std::string &baseName, const TimeSet *refts) const
{
    std::vector<std::string> ret;
    auto allTimes = getAllRealTimes();
    if (!refts) {
        for (size_t i = 0; i < allTimes.size(); i++) {
            ret.push_back(baseName);
        }
        return ret;
    }

    // we may have transient data
    // this is the timeset index of the geometry file
    // name of geometry file
    std::vector<std::string> allGeoFiles;
    auto times = refts->getRealTimes();
    std::vector<std::string> ff = refts->getFileNames(baseName);
    auto tit = times.begin();
    auto fit = ff.begin();
    assert(ff.size() == times.size());
    assert(!times.empty());
    std::string file = *fit;
    for (auto t: allTimes) {
        allGeoFiles.push_back(file);
        if (tit != times.end() && *tit <= t) {
            file = *fit;
            ++fit;
            ++tit;
        }
    }
    return allGeoFiles;
}

const TimeSets &CaseFile::getAllTimeSets() const
{
    return timeSets_;
}

//
// Destructor
//
CaseFile::~CaseFile()
{}

/////////////////////////// class TimeSet //////////////////////////////

TimeSet::TimeSet(const int &n, const int &numTs): num_(n), numTs_(numTs)
{}

TimeSet::TimeSet(const TimeSet &ts): num_(ts.num_), numTs_(ts.numTs_), fileNums_(ts.fileNums_), rTimes_(ts.rTimes_)
{}

int TimeSet::getIdx() const
{
    return num_;
}

void TimeSet::addFileNr(const int &fn)
{
    fileNums_.push_back(fn);
}

void TimeSet::addRealTimeVal(const float &rt)
{
    rTimes_.push_back(rt);
}

int TimeSet::getNumTs() const
{
    return numTs_;
}

std::vector<std::string> TimeSet::getFileNames(const std::string &t) const
{
    // take care of ensight filename convention for time-dependent data
    std::vector<std::string> ret;

    size_t beg(t.find_first_of("*"));
    // return input name if NO asterix is found
    if (beg == std::string::npos) {
        ret.push_back(t);
        return ret;
    }

    size_t end(t.find_first_not_of("*", beg));
    if (end == std::string::npos) {
        end = t.size();
    }

    std::string pre(t.substr(0, beg));
    std::string stars(t.substr(beg, end - beg));
    std::string post(t.substr(end));

    int width(stars.size());

    char ch[64]; // that's more then enough
    for (auto it = fileNums_.begin(); it != fileNums_.end(); it++) {
        sprintf(ch, "%0*d", width, *it);
        std::string fname(pre + std::string(ch) + post);
        ret.push_back(fname);
    }

    return ret;
}

std::vector<float> TimeSet::getRealTimes() const
{
    return rTimes_;
}

std::vector<std::pair<float, std::string>> TimeSet::getTimesAndFileNames(const std::string &t) const
{
    auto rts = getRealTimes();
    auto names = getFileNames(t);
    assert(rts.size() == names.size());
    auto begins = std::make_pair(rts.begin(), names.begin());
    auto ends = std::make_pair(rts.end(), names.end());
    auto result = std::vector<std::pair<float, std::string>>();
    for (auto its = begins; its != ends; its = std::make_pair(std::next(its.first), std::next(its.second))) {
        result.emplace_back(*its.first, *its.second);
    }
    return result;
}

void CaseFile::setBinType(BinType bt)
{
    binType_ = bt;
}

CaseFile::BinType CaseFile::getBinType() const
{
    return binType_;
}
