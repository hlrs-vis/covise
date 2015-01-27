/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: Object modelling ENSIGHT (C) case file                 ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 11.07.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef EN_CASE_FILE_H
#define EN_CASE_FILE_H

#include <util/coviseCompat.h>
#include <map>
using namespace std;

typedef enum
{
    V5,
    V6,
    GOLD
} EnsightVersion;
typedef enum
{
    PER_CELL,
    PER_NODE,
    MYERROR
} DataMode;
typedef map<string, DataMode> DataModeMap;

typedef vector<string> EnFiles;

const int bufLen = 512;

class CaseFile
{
public:
    CaseFile();
    CaseFile(const string &fileNam);

    // true is case file exists
    bool there() const
    {
        return there_;
    };
    // return name of the geo-file
    EnFiles getGeoFiles() const
    {
        return geoFiles_;
    };
    // 	return the number of timesteps in the model
    int getNumTimesteps() const
    {
        return numTimesteps_;
    };
    // return ENSIGHT version
    EnsightVersion getVersion() const
    {
        return enVersion_;
    };
    // return data mode for a file
    // may be used to find out if a given filename belongs to a case file as variable
    DataMode getDataMode(const string &file) const;

    ~CaseFile();

private:
    // we don't want to copy case-files
    CaseFile(const CaseFile & /*cf*/){};
    // no assign
    CaseFile operator=(const CaseFile &cf)
    {
        return cf;
    };

    // read the case-file into  content_;
    void read(const string &fileNm);
    // parse the case file and erase content_;
    void parse();
    // return the basename if str is in ****-notation and str else
    string filterWildcard(const string &str) const;

    vector<string> content_;
    bool there_;
    EnFiles geoFiles_;
    int numTimesteps_;
    EnsightVersion enVersion_;

    DataModeMap dataModes_;
};
#endif
