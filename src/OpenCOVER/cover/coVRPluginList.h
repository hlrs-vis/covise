/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVR_PLUGIN_LIST_H
#define COVR_PLUGIN_LIST_H

/*! \file
 \brief  manage plugins

 \author
 \author (C)
         HLRS, University of Stuttgart,
         Nobelstrasse 19,
         70569 Stuttgart,
         Germany

 \date
 */

#include "coVRDynLib.h"
#include <osg/Drawable>
#include <map>

namespace covise
{
class Message;
};

namespace osg
{
class Node;
class Group;
};

namespace opencover
{
class coVRPlugin;
class RenderObject;
class coInteractor;

class COVEREXPORT coVRPluginList
{
    friend class coVRPluginSupport;
    friend class OpenCOVER;

public:
    // plugin management functions
    //! singleton
    static coVRPluginList *instance();
    //! returns the plugin called name
    coVRPlugin *getPlugin(const char *name) const;

    //! load a plugin, call init, add to list of managed plugins
    coVRPlugin *addPlugin(const char *name);

    //! mark plugin for unloading
    void unload(coVRPlugin *m);

    //
    // methods forwarded to plugins
    //! call addNode method of all plugins
    void addNode(osg::Node *, RenderObject *o = NULL, coVRPlugin *addingPlugin = NULL) const;
    //! call addObject method of all plugins
    void addObject(RenderObject *baseObj,
                   RenderObject *geomObj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   osg::Group *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn,
                   float transparency) const;

    //! call newInteractor method of all plugins
    void newInteractor(RenderObject *container, coInteractor *it) const;

    //! call coviseError method of all plugins
    void coviseError(const char *error) const;

    //! call guiToRenderMsg method of all plugins
    void guiToRenderMsg(const char *msg) const;

    //! call removeObject method of all plugins
    void removeObject(const char *objName, bool replaceFlag) const;
    //! call removeNode method of all plugins
    void removeNode(osg::Node *node, bool isGroup = false, osg::Node *realNode = NULL) const;
    //! call prepareFrame method of all plugins
    void prepareFrame() const;
    //! call preFrame method of all plugins
    void preFrame();
    //! call postFrame method of all plugins
    void postFrame() const;
    //! call preDraw method of all plugins
    void preDraw(osg::RenderInfo &renderInfo) const;
    //! call preSwapBuffers method of all plugins
    void preSwapBuffers(int windowNumber) const;
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
    void setTimestep(int timestep) const;
    //! send a message to all plugins
    void message(int t, int l, const void *b) const;
    //! add new plugins, if not already loaded
    //! unpack and distribute a Message
    void forwardMessage(int len, const void *buf) const;
    //! request to terminate COVER or COVISE session
    void requestQuit(bool killSession) const;
    //! send a message to COVISE/visualisation system - delivered via only one plugin
    bool sendVisMessage(const covise::Message *msg) const;
    //! request to become master of a collaborative session - return true if delivered
    bool becomeCollaborativeMaster() const;
    //! for visualisation system plugins: wait for message, return NULL if no such plugin
    covise::Message *waitForVisMessage(int messageType) const;
    //! for visualisation system plugins: execute data flow network - return true if delivered
    bool executeAll() const;
    //! allow plugins to expand scene bounding sphere
    void expandBoundingSphere(osg::BoundingSphere &bs) const;

    //! called by plugin's commitTimestep method when timestep is prepared
    void commitTimestep(int t, coVRPlugin *caller);

private:
    coVRPluginList();
    ~coVRPluginList();
    //! call init method of all plugins
    void init();
    //! unload all plugins queued for unloading
    void unloadQueued();

    void unloadAllPlugins();
    //! try to load a plugin
    coVRPlugin *loadPlugin(const char *name);

    void grabKeyboard(coVRPlugin *);
    coVRPlugin *keyboardGrabber() const
    {
        return keyboardPlugin;
    }
    int pointerGrabbed;
    coVRPlugin *keyboardPlugin;
    static coVRPluginList *singleton;
    void updateState();

    typedef std::map<std::string, coVRPlugin *> PluginMap;
    PluginMap m_plugins;

    typedef std::vector<CO_SHLIB_HANDLE> HandleList;
    HandleList m_unloadNext, m_unloadQueue;
    int m_numOutstandingTimestepPlugins;
    int m_requestedTimestep;
};
}
#endif
