#ifndef VISTLE_READENSIGHT_CASEFILE_H
#define VISTLE_READENSIGHT_CASEFILE_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    CaseFile
//
// Description: Representation of EnSight case file
//
// Initial version: 2002-
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#include "DataItem.h"

#include <vector>
#include <map>
#include <string>
#include <utility>

class TimeSet;

typedef std::map<std::string, DataItem> DataMap;
typedef std::vector<TimeSet *> TimeSets;

class CaseFile {
public:
    enum { v5, v6, gold };
    enum BinType { CBIN, FBIN, NOBIN, UNKNOWN };

    /// default CONSTRUCTOR
    CaseFile();

    // live clean and easy: copy constructor
    CaseFile(const CaseFile &cf);

    /// DESTRUCTOR
    ~CaseFile();

    void addDataItem(const DataItem &it);
    void setGeoFileNm(const std::string &fn);
    void setMGeoFileNm(const std::string &fn);
    void setGeoTsIdx(const int &idx);
    int getGeoTsIdx() const;
    void setVersion(const int &v);
    void setBinType(BinType bt);
    void setConnectivityFileIndex(int idx);
    int getConnectivityFileIndex() const;
    BinType getBinType() const;

    // set the full name of the case file
    void setFullFilename(const std::string &fn);

    const DataMap &getDataIts() const { return dataIts_; };
    const DataItem *getDataItem(const std::string &filename) const;

    std::string getGeoFileNm() const;
    std::string getMGeoFileNm() const;

    // returns the directory in which the case file is found
    std::string getDir() const;

    // returns the name of the case without extension
    std::string getProjectName();

    std::string printEnVersion();

    int getVersion() const { return version_; };

    bool empty() const { return empty_; };

    void addTimeSet(TimeSet *ts);
    const TimeSets &getAllTimeSets() const;
    const TimeSet *getLastTimeSet() const;
    const TimeSet *getTimeSet(int idx) const;
    const TimeSet *getTimeSet(const std::string &field) const;
    const TimeSet *getGeoTimeSet() const;
    TimeSet *getLastTimeSet();
    std::vector<float> getAllRealTimes() const;

    std::vector<std::string> makeFileNames(const std::string &baseName, const TimeSet *ts) const;

private:
    bool empty_;
    std::string geoFileNm_;
    std::string mgeoFileNm_;
    DataMap dataIts_;
    int version_;
    std::string fullFilename_;
    std::string dir_;
    std::string projectNm_;
    TimeSets timeSets_;
    std::map<std::string, int> fieldMap_;
    int geoTsIdx_; // time set index of geometry
    BinType binType_ = UNKNOWN;
    int connectivityFileIndex_ = -1; // index of the connectivity file, -1: all time steps
};

// simple data class to store time step information
class TimeSet {
public:
    TimeSet(const int &n, const int &numTs);
    TimeSet(const TimeSet &ts);

    // add item (fileNr., realTime)
    void add(const int &fn, const float &rt);

    void addFileNr(const int &fn);
    void addRealTimeVal(const float &fn);

    int getNumTs() const;
    int getIdx() const;

    // crate array of filenames to read from a givene template in ensight
    // filename convention (asteriks)
    std::vector<std::string> getFileNames(const std::string &templ) const;
    std::vector<std::pair<float, std::string>> getTimesAndFileNames(const std::string &t) const;

    // get all real time values from case-file
    std::vector<float> getRealTimes() const;

    int size() const { return (int)fileNums_.size(); }

private:
    int num_; // number of *this
    int numTs_; // number of timesteps
    std::vector<int> fileNums_; // file numbers
    std::vector<float> rTimes_; // real times
};
#endif
