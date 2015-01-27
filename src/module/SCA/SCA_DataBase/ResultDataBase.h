/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _RESULT_DATA_BASE_H
#define _RESULT_DATA_BASE_H

#include <cstdlib>
#include <cstdio>
#include <util/coExport.h>

#include "ResultParam.h"

#include <vector>
#include <string>

using std::string;

class SCAEXPORT Candidate
{
public:
    Candidate(const char *wholepath, const char *path, ResultParam *param,
              const Candidate *, const float *minmax, float error);
    const char *getPath();
    float getDiff() const;
    std::vector<float> &getDiffArray();
    string &getWholePath();

private:
    string wholepath_;
    string path_; // this is redundant but convenient...
    float diff_;
    std::vector<float> diffArray_;
};

class SCAEXPORT ResultDataBase
{
public:
    ResultDataBase(const char *path);
    ~ResultDataBase()
    {
        delete[] path_;
    };

    const char *getSaveDirName(int num, std::vector<ResultParam *> &p_list);
    const char *searchForResult(float &diff, std::vector<ResultParam *> &p_list,
                                std::vector<Candidate *> &FinalCandidates,
                                float error,
                                int limit_level = NUM_PARAMS);

private:
    static const int NUM_PARAMS = 10;
    int searchInLevel(int level,
                      std::vector<ResultParam *> &p_list,
                      std::vector<Candidate *> &Candidates,
                      std::vector<Candidate *> &FinalCandidates,
                      float error,
                      int limit_level = NUM_PARAMS);
    const char *DIR_SEP;
    char *path_;
    char *directory_;
};
#endif
