/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++                                                                     ++
// ++ Description: read, compares, and writes an update COVISE net file   ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 16.01.2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#ifndef NETFILE_H
#define NETFILE_H

#include "Skeletons.h"
#include "Connection.h"

#include <iostream>
#include <string>
#include <vector>
using namespace std;

//
// type to be used to stear the NetFile::check method
//
// AUTO_REM  : remove modules if not present in current COVISE installation
// QUERY_REM : query user if modules should be removed
// NO_REM    : don't remove modules at all; use the module information form the file to be
//             converted instead
//
typedef enum
{
    AUTO_REM,
    QUERY_REM,
    NO_REM
} CheckMode;

class NetFile
{
public:
    NetFile();
    NetFile(Skeletons *skels, ostream &sDiag = cerr);

    int read(const std::string &filename);

    void check(CheckMode mode);

    ~NetFile();

    int getNumModules() const;

    // array will contain the num names of the modules found in
    // NetFile
    void getNameList(std::vector<std::string> &array, int &n);

    // replaces user and/or host fields
    // the argument has to have the following form: "olduser@oldhost:newuser@newhost"
    // if olduser OR oldhost is not found the method returns 0; 1 else
    int replaceUserHost(const std::string &userHostStr);

    // replaces all remote hosts by LUSER/LHOST
    void makeLocal();

    std::string getInputFileName() const;

    // write the converted output to s
    friend ostream &operator<<(ostream &s, const NetFile &f);

protected:
    // add to the internal arrays
    void add(const ModuleSkeleton &mod);
    void add(const Connection &con);
    void add(const std::string &host, const std::string &user);

    const ModuleSkeleton &get(const std::string &name, const int &idx);

    // count all modules in current map and renumber the modules
    void reCnt();

    int fileRead_;

    ostream &sDiag_;

    Skeletons *skels_;
    std::vector<std::string> inLines_;
    //std::string *inLines_;
    //int inLinesLen_;
    int allocInc_;
    int allocSize_;

    // arrays for host/user entries
    std::string *hosts_;
    std::string *users_;
    int numHosts_;

    std::vector<ModuleSkeleton> modules_; // array of used modules
    ModuleSkeleton emptyMod_;

    std::vector<Connection> connections_; // array of Connections

    std::string inputFileName_;
    int version_; // net file version, 0 if not present
};
#endif
