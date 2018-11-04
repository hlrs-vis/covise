/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description: Skeletons stores and obtains module data               ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 11.01.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef COSKELETONS_H
#define COSKELETONS_H

#include "ModuleSkel.h"
#include "Translations.h"
#include <vector>

const int SUCCESS = 1;
const int FAILURE = 0;

class Skeletons
{
public:
    Skeletons();

    Skeletons(const Skeletons &rs);

    // obtains module skeletons by querying modules
    int obtainLocal(const char *bDir);

    // read translation table
    // returns FAILURE if file could not opened; SUCCESS else
    int getTranslations(const char *file);

    // sets an array of known module-names
    // if the name list is set (n>0) only those modules are queried
    // which are contain in the name list
    void setNameList(const std::vector<std::string> &nmeLst)
    {
        nameList_ = nmeLst;
    }

    // compares name-list with translations
    void normalizeNameList();

    int nameListLen()
    {
        return (int)nameList_.size();
    }

    // adds a module skeleton
    int add(const ModuleSkeleton &skel);

    // returns the number of known module skeletons
    int getNumSkels()
    {
        return (int)skels_.size();
    }

    // returns a module skeleton by name
    const ModuleSkeleton &get(const char *name, const char *grp);

    // acticate caching (T.B.D.)
    void acticateCache(){};

    // deactivate caching (T.B.D.)
    void deactivateCache(){};

    // returns 1 if caching is activated (T.B.D.)
    int cacheActive() const
    {
        return 0;
    };

    virtual ~Skeletons();

    const std::string *findReplacement(std::string name);

private:
    // assignment is explicitly forbidden
    const Skeletons &operator=(const Skeletons &rs);

    // write module information to a cache file (T.B.D.)
    void writeToCache();

    // read module from cache (T.B.D.)
    void readCache();

    // return 1 if cache is recent; -1 if unreadable (T.B.D.)
    int actCache();

    // parse output of modules and create the skeletons array
    int parseModOutput(const std::string &group, const std::vector<std::string> &lines, const std::string &name);
    // remove name from nameList_
    int remFromNameList(const std::string &name);

    std::vector<ModuleSkeleton> skels_;
    ModuleSkeleton emptySkel_; // dummy for error conditions

    std::vector<std::string> nameList_;
    Translations nameAliases_;
    Translations grpAliases_;
    Translations multPrmAliases_; // tanslations for <parameter>@<module>
    Translations multPortAliases_; // tanslations for <port>@<module>
    int isNormalized_;
};
#endif
