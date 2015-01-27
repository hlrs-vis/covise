/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    ModuleInfo
//
// Description: data class singleton to proper globalize module specific
//              information
//
// Initial version: 2001-12-20
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef MODULEINFO_H
#define MODULEINFO_H

#include <string>

class ModuleInfoSingleton
{
public:
    // the one and only access
    static ModuleInfoSingleton *instance();

    void initialize(const std::string &hostname, const int &inst);

    std::string getHostname();
    int iGetInstance();
    std::string getInstance();
    std::string getCoMsgHeader();

    // if called the execution of the whole programm will be stoped
    // after accessing the uninitialzed object
    void setExitOnUninit()
    {
        doExit_ = true;
    };

    /// DESTRUCTOR
    ~ModuleInfoSingleton();

private:
    /// default CONSTRUCTOR
    ModuleInfoSingleton();

    // !! helper: exit if not initialized
    void initCheck();

    bool initialized_;
    bool doExit_;

    std::string moduleName_;
    std::string hostname_;
    int modInstance_;

    static ModuleInfoSingleton *instance_;
};

// global variable to simplify the access to the one and only instance
static ModuleInfoSingleton *ModuleInfo = ModuleInfoSingleton::instance();
#endif
