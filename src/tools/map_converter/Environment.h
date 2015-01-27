/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    Environment
//
// Description: Singleton to get environment variables
//
// Initial version: 15.11.2001 rm
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//
//
//

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

//
// system includes
//
#include <cstdlib>
#include <vector>
#include <string>

typedef std::vector<std::string> PathList;

/// Singleton to access environment variables.
/// It uses getenv therefore an initialzation is not required.
class EnvSingleton
{
public:
    /// the one and only access

    /// @return        pointer to the one and only instance
    static EnvSingleton *instance();

    /// get environment variable

    /// @param   name of the variable

    /// @return  value of the env. variable, empty if variable does not exist or has no value
    std::string get(const std::string &name) const;

    /// shows if env variable exists

    /// @param   name of the variable

    /// @return  true if variable exists
    bool exist(const std::string &name) const;

    /// The method scans a variable value to extract parts which are separated by a
    /// given delimiter. Made for variables of the Unix PATH type

    ///  @param varName  name of the variable to scan

    ///  @param del      std::string containing the delimiter

    ///  @return         list of entries
    PathList scan(const std::string &name, const std::string &del = std::string(":")) const;

    /// DESTRUCTOR
    ~EnvSingleton();

private:
    /// default CONSTRUCTOR we are are Singleton therefore private
    EnvSingleton();
    /// assignment is explicitly forbidden
    const EnvSingleton &operator=(const EnvSingleton &es);

    static EnvSingleton *instance_;
};

// global variable to simlify the access to the one and only instance
static const EnvSingleton *Environment = EnvSingleton::instance();
#endif
