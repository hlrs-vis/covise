/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class EnvSingleton                    ++
// ++                                                                     ++
// ++ Author (initial version):  Ralf Mikulla (rm@vircinity.com)          ++
// ++                                                                     ++
// ++                            VirCinity GmbH                           ++
// ++                            Nobelstrasse 15                          ++
// ++                            70569 Stuttgart                          ++
// ++                                                                     ++
// ++ Date: 11-15-2001                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Environment.h"
#include <string>
#include <util/string_util.h>

EnvSingleton *EnvSingleton::instance_ = NULL;

//
// Constructor
//
EnvSingleton::EnvSingleton()
{
}

//
//  access method
//
EnvSingleton *
EnvSingleton::instance()
{
    if (!instance_)
    {
        instance_ = new EnvSingleton;
    }
    return instance_;
}

//
// return value of environment variable by name
//
std::string
EnvSingleton::get(const std::string &name) const
{
    std::string retStr;
    char *environment = getenv(name.c_str());
    if (environment != NULL)
    {
        retStr = std::string(environment);
    }
    return retStr;
}

//
// return true if variable exists
//
bool
EnvSingleton::exist(const std::string &name) const
{
    bool ret = false;
    char *environment = getenv(name.c_str());
    if (environment != NULL)
        ret = true;

    return ret;
}

//
// scane env-variable
//
PathList
EnvSingleton::scan(const std::string &name, const std::string &del) const
{
    std::string val;

    PathList retVal;
    // getting the einvironment variable by calling getenv.
    // Return if del is empty, or the value returned by getenv
    // is empty.
    if (del.empty())
        return retVal;
    char *environment = getenv(name.c_str());
    if (environment != NULL)
    {
        val = std::string(environment);
    }
    else
    {
        return retVal;
    }

    // now we have it and scan the variable
    //    int idx = val.find_first_of(del.c_str());
    size_t idx = 0;
    size_t start = 0;
    // adding spaces at the beginning and end of val is a workaround
    // of a flaw in std::string::find_first_of(..) and will be checked later
    std::string sVal(" ");
    sVal = sVal + val;
    sVal = sVal + std::string(" ");
    // we ignore a leading del
    while (idx != std::string::npos)
    {
        idx = sVal.find_first_of(del.c_str(), idx + 1);
        if (idx == std::string::npos)
        {
            size_t end = sVal.size();
            std::string part = strip(sVal.substr(start, end - start));
            if (!part.empty())
                retVal.push_back(part);
        }
        else
        {
            std::string part = strip(sVal.substr(start, idx - start));
            if (!part.empty())
                retVal.push_back(part);
            start = idx + 1;
        }
    }

    return retVal;
}

//
// Destructor
//
EnvSingleton::~EnvSingleton()
{
    delete instance_;
}

//************************************ TESTING ***********************************
//
// use the following command to test the class
// setenv C++ <your c++ compiler>
// $C++ -DTESTING -g -I$COVISEDIR/src/controller/ Environment.cpp $COVISEDIR/objects_linux/CoNewString.o
//
// and then type ./a.out
//
//************************************ TESTING ***********************************
#ifdef TESTING
int
main()
{

    PathList pl = Environment->scan("PATH", ":");

    if (pl.empty())
        cerr << "PATHLIST is empty" << endl;

    for (int i = 0; i < pl.size(); ++i)
    {
        cerr << "PART :" << pl[i] << endl;
    }
}
#endif
