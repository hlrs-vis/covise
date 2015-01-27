/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef HPPARCOMP_PLUGIN_H
#define HPPARCOMP_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2006 HLRS  **
 **                                                                          **
 ** Description: Parallel Rendering using the HP parallel compositing        **
 ** libary                                                                   **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRParallelRenderPlugin.h>

#include "SortLastImplementation.h"

class SortLastHPParComp : public opencover::coVRParallelRenderPlugin
{
public:
    SortLastHPParComp();
    ~SortLastHPParComp();

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
