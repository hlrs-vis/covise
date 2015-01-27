/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SORTLASTIMPLEMENTATION_H
#define SORTLASTIMPLEMENTATION_H

#include <string>
#include <list>

class SortLastImplementation
{
public:
    SortLastImplementation(const std::string &nodename, int session)
        : nodename(nodename)
        , session(session)
    {
    }
    virtual ~SortLastImplementation()
    {
    }

    virtual void preFrame()
    {
    }
    virtual void postFrame()
    {
    }
    virtual bool init()
    {
        return true;
    }
    virtual void preSwapBuffers(int windowNumber)
    {
        (void)windowNumber;
    }

    virtual bool initialiseAsMaster() = 0;
    virtual bool initialiseAsSlave() = 0;
    virtual bool createContext(const std::list<std::string> &hostlist, int groupIdentifier) = 0;

protected:
    std::string nodename;
    int session;
};

#endif // SORTLASTIMPLEMENTATION_H
