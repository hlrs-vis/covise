/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SORTLAST_PLUGIN_H
#define SORTLAST_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2011 HLRS  **
 **                                                                          **
 ** Description: Parallel Rendering using the Sort Last                      **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRParallelRenderPlugin.h>

#include "SortLastImplementation.h"

class SortLast : public opencover::coVRParallelRenderPlugin
{
public:
    SortLast();
    ~SortLast();

    virtual void preFrame();
    virtual void postFrame();
    virtual bool init();
    virtual void preSwapBuffers(int windowNumber);

    virtual std::string getHostID() const;
    virtual bool initialiseAsMaster();
    virtual bool initialiseAsSlave();
    virtual bool createContext(const std::list<std::string> &hostlist, int groupIdentifier);

private:
    bool isMaster;
    std::string nodename;
    int session;

    SortLastImplementation *impl;
};
#endif
