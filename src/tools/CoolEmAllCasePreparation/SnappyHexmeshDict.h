/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SnappyHexmeshDict_H
#define SnappyHexmeshDict_H

#include <string>
#include <iostream>
#include <fstream>
#include <list>
#include <set>
#include "FileReference.h"

class CoolEmAll;

class PatchInfo
{
public:
    std::string keyword;
    std::string patchName;
    std::string databasePath;
};

class SnappyHexmeshDict
{
public:
    SnappyHexmeshDict(CoolEmAll *cc);
    ~SnappyHexmeshDict();
    void writeHeader();
    void writeSTL(std::string DataBase_Path, FileReference *ProductRevisionViewReference, FileReference *ProductInstanceReference, std::string transformedSTLFileName);
    void writeFooter();

private:
    std::ofstream file1;
    CoolEmAll *cool;
    std::list<std::string> patches;
    std::list<std::string>::iterator patchesIt;
    std::list<std::string> DEBBLevels;
    std::list<PatchInfo> patchAverageList;
    std::list<PatchInfo>::iterator it;
    std::string locationinmesh;
};

#endif
