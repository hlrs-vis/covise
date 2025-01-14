/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include "vvDynLib.h"
#include "vvPlugin.h"
#include <vrb/client/SharedState.h>
#include <map>

namespace vrb
{
    class UdpMessage;
}

namespace covise
{
class Message;
class DataHandle;
}


namespace vive
{
class vvRenderObject;
class vvInteractor;

class VVCORE_EXPORT vvPluginList
{
    friend class vvPluginSupport;
    friend class OpenCOVER;

public:
    enum PluginDomain
    {
        Default,
        Vis,
        Window,
        Input,
        NumPluginDomains // keep last
    };

    // plugin management functions
    //! singleton
    static vvPluginList *instance();
    //! load configured plugins
    void loadDefault();

    //! returns the plugin called name
    vvPlugin *getPlugin(const char *name) const;

    //! load a plugin, call init, add to list of managed plugins
    vvPlugin *addPlugin(const char *name, PluginDomain domain=Default);

    //! mark plugin for unloading
    void unload(vvPlugin *m);

    //
    // methods forwarded to plugins


    //! call addNode method of all plugins
    void addNode(vsg::Node *, const vvRenderObject *o = NULL, vvPlugin *addingPlugin = NULL) const;

    //! call addObject method of all plugins
    void addObject(const vvRenderObject *container, vsg::Group *root,
            const vvRenderObject *geometry, const vvRenderObject *normals, const vvRenderObject *colors, const vvRenderObject *texture) const;

    //! call newInteractor method of all plugins
    void newInteractor(const vvRenderObject *container, vvInteractor *it) const;

    //! call enableInteraction method of all plugins until one is accepting the request
    bool requestInteraction(vvInteractor *inter, vsg::Node *triggerNode, bool isMouse);

    //! call coviseError method of all plugins
    void coviseError(const char *error) const;

    //! call guiToRenderMsg method of all plugins
    void guiToRenderMsg(const grmsg::coGRMsg &msg)  const;

    //! call removeObject method of all plugins
    void removeObject(const char *objName, bool replaceFlag) const;
    //! call removeNode method of all plugins
    void removeNode(vsg::Node *node, bool isGroup = false, vsg::Node *realNode = NULL) const;
    //! call update method of all plugins
    bool update() const;
    //! call preFrame method of all plugins
    void preFrame();
    //! call postFrame method of all plugins
    void postFrame() const;
    //! call preDraw method of all plugins
    void preDraw() const;
    //! call preSwapBuffers method of all plugins
    void preSwapBuffers(int windowNumber) const;
    //! call clusterSyncDraw() method of all plugins
    void clusterSyncDraw() const;
    //! call postSwapBuffers method of all plugins
    void postSwapBuffers(int windowNumber) const;
    //! call param method of all plugins
    void param(const char *paramName, bool inMapLoading) const;
    //! call key method of all plugins
    bool key(int type, int keySym, int mod) const;
    //! call userEvent method of all plugins
    bool userEvent(int mod) const;
    //! call requestTimestep method of all plugins
    void requestTimestep(int timestep);
    //! call setTimestep method of all plugins
    void setTimestep(int timestep);
    //! send a message to all plugins
    void message(int toWhom, int t, int l, const void *b, const vvPlugin *exclude = nullptr) const;
    //! send a UDPmessage to all plugins
	void UDPmessage(covise::UdpMessage* msg) const;
    //! add new plugins, if not already loaded
    //! unpack and distribute a Message
    void forwardMessage(const covise::DataHandle &dh) const;
    //! request to terminate COVER or COVISE session
    void requestQuit(bool killSession) const;
    //! send a message to COVISE/visualisation system - delivered via only one plugin
    bool sendVisMessage(const covise::Message *msg) const;
    //! request to become master of a collaborative session - return true if delivered
    bool becomeCollaborativeMaster() const;
    //! for visualisation system plugins: wait for message, return NULL if no such plugin
    covise::Message *waitForVisMessage(int messageType) const;
    //! allow plugins to expand scene bounding sphere
    //void expandBoundingSphere(vsg::BoundingSphere &bs) const;

    //! called by plugin's commitTimestep method when timestep is prepared
    void commitTimestep(int t, vvPlugin *caller);

    void unloadAllPlugins(PluginDomain domain=Default);

    //! call init method of all plugins
    void init();
    //! call init2 method of all plugins
    void init2();
private:
    vvPluginList();
    ~vvPluginList();
    vvPluginList(const vvPluginList &) = delete;
    vvPluginList(vvPluginList &&) = delete;
    vvPluginList &operator=(const vvPluginList &) = delete;
    vvPluginList &operator=(vvPluginList &&) = delete;
    //! unload all plugins queued for unloading
    void unloadQueued();
    //! add plugin to list of managed plugins
    void manage(vvPlugin *plug, PluginDomain domain);
    //! remove plugin from list of managed plugins
    void unmanage(vvPlugin *plug);

    //! try to load a plugin
    vvPlugin *loadPlugin(const char *name, bool showErrors = true);

    void grabKeyboard(vvPlugin *);
    vvPlugin *keyboardGrabber() const
    {
        return keyboardPlugin;
    }
    int pointerGrabbed;
    vvPlugin *keyboardPlugin;

    void grabViewer(vvPlugin *);
    vvPlugin *viewerGrabber() const;
    vvPlugin *viewerPlugin = nullptr;

    static vvPluginList *singleton;
    void updateState();

    typedef std::map<std::string, vvPlugin *> PluginMap;
    PluginMap m_plugins;
    std::vector<vvPlugin *> m_loadedPlugins[NumPluginDomains];
    std::map<std::string, std::unique_ptr<vrb::SharedState<bool>>> m_sharedPlugins;
    typedef std::vector<CO_SHLIB_HANDLE> HandleList;
    HandleList m_unloadNext, m_unloadQueue;
    int m_numOutstandingTimestepPlugins = 0;
    int m_requestedTimestep = -1;
    int m_currentTimestep = 0;
    mutable bool m_stopIteration = false;
};
}
