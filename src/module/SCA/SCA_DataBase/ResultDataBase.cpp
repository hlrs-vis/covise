/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                    (C) 2002 VirCinity  **
 ** Author: Sven Kufer		                                          **
 **         (C)  VirCinity IT- Consulting GmbH                             **
 **         Nobelstrasse 15                                                **
 **         D- 70569 Stuttgart    			       	          **
 **                                                                        **
 **  Date:22.7.2002                                                         **
 **                                                                        **
\**************************************************************************/

#include "ResultDataBase.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

#ifdef __sgi
#include <sys/dir.h>
#else
#include <dirent.h>
#endif

ResultDataBase::ResultDataBase(const char *path)
{
    path_ = new char[strlen(path) + 1];
    strcpy(path_, path);
#ifndef WIN32
    DIR_SEP = "/";
#else
    DIR_SEP = "\\";
#endif
    directory_ = NULL;
}

const char *
ResultDataBase::getSaveDirName(int num, std::vector<ResultParam *> &p_list)
{
    int size = 0;
    int i;

    delete[] directory_;

    for (i = 0; i < num; i++)
    {
        size += strlen(p_list[i]->getDirName()) + strlen(DIR_SEP);
    }

    size += strlen(path_) + 1;
    directory_ = new char[size];

    struct stat existDir;

    strcpy(directory_, path_);
    for (i = 0; i < num; i++)
    {
        strcat(directory_, p_list[i]->getDirName());
        /// create dir if it doesn't exist
        if (stat(directory_, &existDir) == -1)
        {
            if (errno == ENOENT)
            {
#ifdef _WIN32
                mkdir(directory_);
#else
                mkdir(directory_, S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
#endif
            }
        }
        strcat(directory_, DIR_SEP);
    }

    return directory_;
}

std::vector<float> &
Candidate::getDiffArray()
{
    return diffArray_;
}

string &
Candidate::getWholePath()
{
    return wholepath_;
}

const char *
Candidate::getPath()
{
    return wholepath_.c_str();
}

float
Candidate::getDiff() const
{
    return diff_;
}

#include <float.h>
#include <assert.h>

Candidate::Candidate(const char *wholepath,
                     const char *path,
                     ResultParam *param,
                     const Candidate *prevCand,
                     const float *minmax,
                     float error)
    : // highest tolerated error
    wholepath_(wholepath)
    , path_(path)
{
    if (prevCand)
    {
        diff_ = prevCand->getDiff();
    }
    else
    {
        diff_ = 0.0;
    }

    if (diff_ == FLT_MAX)
    {
        return;
    }

    // get additional diff from param
    const char *paramValue = param->getValue();
    float sollval, istval;

    if (sscanf(paramValue, "%g", &sollval) != 1) // string
    {
        // find '=' in path_
        const char *pathValue = strstr(path_.c_str(), "=");
        if (pathValue == NULL)
        {
            diff_ = FLT_MAX;
            return;
        }
        ++pathValue;
        if (strcmp(paramValue, pathValue) != 0)
        {
            diff_ = FLT_MAX;
        }
    }
    else // number in param
    {
        const char *startNum = strstr(path_.c_str(), "=");
        if (startNum == NULL)
        {
            diff_ = FLT_MAX;
        }
        else
        {
            ++startNum;
            if (sscanf(startNum, "%g", &istval) == 1) // float, compare with istval
            {
                assert(minmax != NULL);
                if (param->getType() != ResultParam::FLOAT)
                {
                    if ((istval - sollval) * (istval - sollval) > error)
                    {
                        diff_ = FLT_MAX;
                        return;
                    }
                }
                float norm = minmax[1] - minmax[0];
                norm *= norm;
                // FIXME
                diff_ += (istval - sollval) * (istval - sollval) / norm;
                // add to diffArray_
                int prev; // in prevCand
                if (prevCand)
                {
                    for (prev = 0; prev < (prevCand->diffArray_).size(); ++prev)
                    {
                        diffArray_.push_back((const_cast<Candidate *>(prevCand)->diffArray_)[prev]);
                    }
                }
                diffArray_.push_back(diff_);
            }
            else
            {
                diff_ = FLT_MAX;
            }
        }
    }
}

#include "ReadASCIIDyna.h"

int
ResultDataBase::searchInLevel(int level,
                              std::vector<ResultParam *> &p_list,
                              std::vector<Candidate *> &Candidates,
                              std::vector<Candidate *> &FinalCandidates,
                              float error,
                              int limit_level)
{
    DIR *dirp;
#ifdef __sgi
    struct direct *entry;
#else
    struct dirent *entry;
#endif

    std::vector<Candidate *> newCandidates; // candidates for the next level

    if (level == 0)
    {
        dirp = opendir(path_);
        if (dirp != NULL)
        {
            while ((entry = readdir(dirp)) != NULL)
            {
                if (strstr(entry->d_name, "=") == NULL)
                {
                    continue;
                }
                string path(path_);
                path += entry->d_name;
                path += DIR_SEP;
                newCandidates.push_back(new Candidate(path.c_str(),
                                                      entry->d_name,
                                                      p_list[level],
                                                      NULL,
                                                      ReadASCIIDyna::getMinMax(level),
                                                      error));
            }
        }
        else
        {
            cerr << "starting database directory" << path_ << " does not exist" << endl;
            return -1;
        }
        closedir(dirp);
    }
    else // level > 0
    {
        int candidate;
        for (candidate = 0; candidate < Candidates.size(); ++candidate)
        {
            const char *debug = Candidates[candidate]->getPath();
            dirp = opendir(debug);
            while ((entry = readdir(dirp)) != NULL)
            {
                if (strstr(entry->d_name, "=") == NULL)
                {
                    continue;
                }
                string path(Candidates[candidate]->getPath());
                path += entry->d_name;
                path += DIR_SEP;
                newCandidates.push_back(new Candidate(path.c_str(),
                                                      entry->d_name,
                                                      p_list[level],
                                                      Candidates[candidate],
                                                      ReadASCIIDyna::getMinMax(level),
                                                      error));
            }
            closedir(dirp);
        }
    }

    if (level == limit_level - 1)
    {
        // OK, we have all final candidates at last in newCandidates
        int candidate;
        for (candidate = 0; candidate < newCandidates.size(); ++candidate)
        {
            FinalCandidates.push_back(newCandidates[candidate]);
        }
    }
    else
    {
        searchInLevel(level + 1, p_list, newCandidates, FinalCandidates, error,
                      limit_level);
        int candidate;
        for (candidate = 0; candidate < newCandidates.size(); ++candidate)
        {
            delete newCandidates[candidate];
        }
    }

    return 0;
}

const char *
ResultDataBase::searchForResult(float &diff,
                                std::vector<ResultParam *> &p_list,
                                std::vector<Candidate *> &FinalCandidates,
                                float error,
                                int limit_level)
{
    diff = FLT_MAX;
    const char *ret = NULL;
    std::vector<Candidate *> Candidates;
    // search over all possible subdirectories...
    // start at level 0...
    searchInLevel(0, p_list, Candidates, FinalCandidates,
                  error, limit_level);
    int fcandidate;
    for (fcandidate = 0; fcandidate < FinalCandidates.size(); ++fcandidate)
    {
        if (FinalCandidates[fcandidate]->getDiff() < diff)
        {
            diff = FinalCandidates[fcandidate]->getDiff();
            ret = FinalCandidates[fcandidate]->getPath();
        }
    }
    // cleaning up (FinalCandidates)
    // FinalCandidates are no longer needed... cleaning up
    /*
      int candidate;
      for(candidate=0;candidate<FinalCandidates.size();++candidate){
         delete FinalCandidates[candidate];
      }
   */
    return ret;
}

/*
const char*
ResultDataBase::searchForResult( float &diff,
                                 int num,
                                 std::vector<ResultParam *> &p_list )
{
   int i, j;

   delete[] directory_;
   directory_ = NULL;

CoString path(path_);

struct stat existDir;

char **entries = new char *[1024];
int num_entries;

DIR *dirp;
#ifdef __sgi
struct direct *entry;
#else
struct dirent *entry;
#endif

for( i=0; i<num; i++ ) {

path += p_list[i]->getDirName();
/// look for nearest if it doesn't exist

if( stat( path.getValue(), &existDir )==-1 ) {
if( errno==ENOENT ) {
num_entries = 0;

path.del(p_list[i]->getDirName());

dirp = opendir(path.getValue());
if (dirp != NULL) {
while ((entry = readdir(dirp)) != NULL) {
entries[num_entries] = new char[strlen(entry->d_name)+1];
strcpy( entries[num_entries++], entry->d_name);
}

float cur_diff=0.0;
const char *thisClosest = p_list[i]->
getClosest( cur_diff, num_entries, entries );
if(thisClosest){
path += thisClosest;
diff += cur_diff;
}
// clean up
for( j=0; j<num_entries; j++ ) {
delete[] entries[j];
}
if(!thisClosest){
return NULL;
}
}
}
else {
return NULL;
}
}
path += DIR_SEP;
}

directory_ = new char[path.length()+1];
strcpy( directory_, path.getValue() );
return directory_;
}
*/
