/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    CaseFile
//
// Description: Representation of ENSIGHT case file
//
// Initial version: 2002-
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef CASEFILE_H
#define CASEFILE_H

#include "DataItem.h"

#include <vector>
#include <string>

class TimeSet;

typedef std::vector<DataItem> DataList;
typedef std::vector<TimeSet *> TimeSets;

// helper: remove leading and trailing spaces
std::string trim(const std::string &s);

class CaseFile
{
public:
    enum
    {
        v5,
        v6,
        gold
    };

    /// default CONSTRUCTOR
    CaseFile();

    // live clean and easy: copy constructor
    CaseFile(const CaseFile &cf);

    /// DESTRUCTOR
    ~CaseFile();

    void addDataIt(const DataItem &it);
    void setGeoFileNm(const std::string &fn);
    void setMGeoFileNm(const std::string &fn);
    void setGeoTsIdx(const int &idx);
    int getGeoTsIdx();
    void setVersion(const int &v);

    // set the full name of the case file
    void setFullFilename(const std::string &fn);

    DataList getDataIts() const
    {
        return dataIts_;
    };

    std::string getGeoFileNm() const;
    std::string getMGeoFileNm() const;

    // returns the directory in which the case file is found
    std::string getDir();

    // returns the name of the case without extension
    std::string getProjectName();

    std::string printEnVersion();

    int getVersion() const
    {
        return version_;
    };

    bool empty() const
    {
        return empty_;
    };

    void addTimeSet(TimeSet *ts);
    const TimeSets &getAllTimeSets();
    TimeSet *getLastTimeSet();

private:
    bool empty_;
    std::string geoFileNm_;
    std::string mgeoFileNm_;
    DataList dataIts_;
    int version_;
    std::string fullFilename_;
    std::string dir_;
    std::string projectNm_;
    TimeSets timeSets_;
    int geoTsIdx_; // time set index of geometry
};

// simple data class to store time step information
class TimeSet
{
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
    std::vector<std::string> getFileNames(const std::string &templ);

    // get all real time values from case-file
    std::vector<float> getRealTimes() const;

    int size() const
    {
        return (int)fileNums_.size();
    };

private:
    int num_; // number of *this
    int numTs_; // number of timesteps
    std::vector<int> fileNums_; // file numbers
    std::vector<float> rTimes_; // real times
};
#endif
