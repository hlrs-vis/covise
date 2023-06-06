/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVRPARALLELRENDERPLUGIN_H
#define COVRPARALLELRENDERPLUGIN_H

#include <cover/coVRPlugin.h>
#include <list>
#include <string>

namespace opencover
{

class coVRParallelRenderPlugin : public coVRPlugin
{
public:
    coVRParallelRenderPlugin(const std::string &name);
    virtual ~coVRParallelRenderPlugin();

    /**
    * Get the host id that can be used to uniquely identify the node.
    */
    virtual std::string getHostID() const = 0;

    /**
    * Called if the plugin should act as display and master of a session.
    * @return true on success
    */
    virtual bool initialiseAsMaster() = 0;

    /**
    * Called if the plugin should act as render slave.
    * @return true on success
    */
    virtual bool initialiseAsSlave() = 0;

    /**
    * Called after initialise to create a context for rendering. Will be called
    * after initialiseAsMaster/Slave and each time the load is rebalanced.
    * @param hostlist A list of hostids. The master is the first entry in this list.
    * @param A group identifier common to a master and its slaves but unique to all others.
    * @return true on successful context creation
    */
    virtual bool createContext(const std::list<std::string> &hostlist, int groupIdentifier) = 0;
};
}
#endif // COVRPARALLELRENDERPLUGIN_H
