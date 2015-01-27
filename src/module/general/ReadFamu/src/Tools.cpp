/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file BuilderGrid.h
 * some generic tools.
 */

#include "Tools.h" // some helpful tools.
#include <math.h>
#include <fstream>

namespace parserTools
{

std::string mid(std::string s, int nFirst)
{
    if ((nFirst > (int)s.length()) || ((int)nFirst < 0))
        return "";

    return s.substr(nFirst, s.length() - nFirst);
}

std::string mid(std::string s, int nFirst, int nCount)
{
    if ((s == "") || (nCount <= 0) || (nFirst < 0))
        return "";

    return s.substr(nFirst, nCount);
}

int find(std::string s, char c)
{
    return s.find(c, 0);
}

int find(std::string &s, std::string &f)
{
    return s.find(f, 0);
}

std::string replace(std::string s, std::string f, std::string r)
{
    unsigned int fLength = f.length();

    int pos = find(s, f);
    if (pos != -1)
    {
        unsigned int sLength = s.length();
        std::string leftString = mid(s, 0, pos);
        std::string rightString = mid(s, pos + fLength, sLength - pos - fLength);
        std::ostringstream stream;
        stream << leftString;
        stream << r;
        stream << rightString;
        s = stream.str();
    }
    return s;
}

std::string getSuffix(const std::string &filename)
{
    int pos;
    pos = parserTools::find(filename, '.');
    if (pos != -1)
    {
        return mid(filename, pos);
    }
    else
        return "";
}
}

void ArrayIo::writeArrayToDisk(
    std::string filename,
    const int x[],
    int size)
{
    std::ofstream f;
    f.open(filename.c_str());
    int k;
    for (k = 0; k < 5; k++)
    {
        f << k << "\t" << x[k] << "\n";
    }
    f << "...\n";
    int n = 0;
    for (k = 5; k < size - 5; k++)
    {
        if ((x[k] != 0) && n < 5)
        {
            f << k << "\t" << x[k] << "\n";
            n++;
        }
    }
    f << "...\n";
    for (k = size - 5; k < size; k++)
    {
        f << k << "\t" << x[k] << "\n";
    }
    f.close();
}

void ArrayIo::writeArrayToDisk(
    std::string filename,
    const float x[],
    int size)
{
    std::ofstream f;
    f.open(filename.c_str());
    int k;
    for (k = 0; k < 5; k++)
    {
        f << k << "\t" << x[k] << "\n";
    }
    f << "...\n";
    int n = 0;
    for (k = 5; k < size - 5; k++)
    {
        if (fabs(x[k]) > 1e-6 && n < 5)
        {
            f << k << "\t" << x[k] << "\n";
            n++;
        }
    }
    f << "...\n";
    for (k = size - 5; k < size; k++)
    {
        f << k << "\t" << x[k] << "\n";
    }
    f.close();
}

void ArrayIo::writeArrayToDisk(
    std::string filename,
    const double x[],
    const double y[],
    const double z[],
    int noOfPoints)
{
    std::ofstream f;
    f.open(filename.c_str());
    f << "-------------------------------------------\n";
    int k;
    for (k = 0; k < 5; k++)
    {
        f << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
    }
    f << "...\n";
    int n = 0;
    for (k = 5; k < noOfPoints - 5; k++)
    {
        if (fabs(x[k]) > 1e-6 && n < 5)
        {
            f << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
            n++;
        }
    }
    f << "...\n";
    for (k = noOfPoints - 5; k < noOfPoints; k++)
    {
        f << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
    }
    f.close();
}

void ArrayIo::writeArrayToDisk(
    std::string filename,
    const double x[],
    int noOfPoints)
{
    std::ofstream f;
    f.open(filename.c_str());

    f << "-------------------------------------------\n";
    int k;
    for (k = 0; k < 5; k++)
    {
        f << k << "\t" << x[k] << "\n";
    }
    f << "...\n";
    int n = 0;
    for (k = 5; k < noOfPoints - 5; k++)
    {
        if (fabs(x[k]) > 1e-6 && n < 5)
        {
            f << k << "\t" << x[k] << "\n";
            n++;
        }
    }
    f << "...\n";
    for (k = noOfPoints - 5; k < noOfPoints; k++)
    {
        f << k << "\t" << x[k] << "\n";
    }
    f.close();
}

void ExportDataType::writeOverview(const coDoFloat *dataObj,
                                   std::string &name,
                                   std::ofstream &fileHandle)
{

    float *valuesArr;
    dataObj->getAddress(&valuesArr);
    int noOfPoints = dataObj->getNumPoints();

    fileHandle << "name = " << name << "\n";
    INT k;
    for (k = 0; k < 5; k++)
    {
        fileHandle << k << "\t" << valuesArr[k] << "\n";
    }
    fileHandle << "...\n";
    int z = 0;
    for (k = 0; k < noOfPoints - 5; k++)
    {
        if (fabs(valuesArr[k]) > 1e-6 && z < 5)
        {
            fileHandle << k << "\t" << valuesArr[k] << "\n";
            z++;
        }
    }
    fileHandle << "...\n";
    for (k = noOfPoints - 5; k < noOfPoints; k++)
    {
        fileHandle << k << "\t" << valuesArr[k] << "\n";
    }
}

void ExportDataType::writeOverview(const coDoVec3 *dataObj,
                                   std::string &name,
                                   std::ofstream &fileHandle)
{
    float *x, *y, *z;
    dataObj->getAddresses(&x, &y, &z);
    int noOfPoints = dataObj->getNumPoints();

    fileHandle << "name = " << name << "\n";
    INT k;
    for (k = 0; k < 5; k++)
    {
        fileHandle << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
    }
    fileHandle << "...\n";
    int n = 0;
    for (k = 0; k < noOfPoints - 5; k++)
    {
        if (fabs(x[k]) > 1e-6 && n < 5)
        {
            fileHandle << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
            n++;
        }
    }
    fileHandle << "...\n";
    for (k = noOfPoints - 5; k < noOfPoints; k++)
    {
        fileHandle << k << "\t" << x[k] << "\t" << y[k] << "\t" << z[k] << "\n";
    }
}
