/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file BuilderGrid.h
 * some generic tools.
 */

//#include "Tools.h"  // some helpful tools.

#ifndef __Tools_h__
#define __Tools_h__

#include <string>
#include <api/coModule.h>
using namespace covise;
#include <do/coDoData.h>

#include "os.h"

namespace parserTools
{
std::string replace(std::string s, std::string f, std::string r);

std::string getSuffix(const std::string &filename);

std::string mid(std::string s, int nFirst);

int find(std::string s, char c);
}

/**
 * Tool class for in-/output of arrays.
 */
class ArrayIo
{
public:
    static void writeArrayToDisk(
        std::string filename,
        const int arr[],
        int size);

    static void writeArrayToDisk(
        std::string filename,
        const float arr[],
        int size);

    static void writeArrayToDisk(
        std::string filename,
        const double x[],
        int noOfPoints);

    static void writeArrayToDisk(
        std::string filename,
        const double x[],
        const double y[],
        const double z[],
        int noOfPoints);
};

// toolbox for debugging of DataTypes
class ExportDataType
{
public:
    ExportDataType(){};
    virtual ~ExportDataType(){};

    static void writeOverview(const coDoFloat *dataObj, std::string &name, std::ofstream &fileHandle);
    static void writeOverview(const coDoVec3 *dataObj, std::string &name, std::ofstream &fileHandle);
};

#endif
