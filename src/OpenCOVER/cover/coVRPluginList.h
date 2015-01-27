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
    coVRPlugin *getPlugin(const char *name);

    //! load a plugin, call init, add to list of managed plugins
    coVRPlugin *addPlugin(const char *name);

    //! mark plugin for unloading
    void unload(coVRPlugin *m);

    //
    // methods forwarded to plugins
    //! call addNode method of all plugins
    void addNode(osg::Node *, RenderObject *o = NULL, coVRPlugin *addingPlugin = NULL);
    //! call addObject method of all plugins
    void addObject(RenderObject *baseObj,
                   RenderObject *geomObj, RenderObject *normObj,
                   RenderObject *colorObj, RenderObject *texObj,
                   osg::Group *root,
                   int numCol, int colorBinding, int colorPacking,
                   float *r, float *g, float *b, int *packedCol,
                   int numNormals, int normalBinding,
                   float *xn, float *yn, float *zn,
                   float transparency);

    //! call newInteractor method of all plugins
    void newInteractor(RenderObject *container, coInteractor *it);

    //! call coviseError method of all plugins
    void coviseError(const char *error);

    //! call guiToRenderMsg method of all plugins
    void guiToRenderMsg(const char *msg);

    //! call removeObject method of all plugins
    void removeObject(const char *objName, bool replaceFlag);
    //! call removeNode method of all plugins
    void removeNode(osg::Node *node, bool isGroup = false, osg::Node *realNode = NULL);
    //! call prepareFrame method of all plugins
    void prepareFrame();
    //! call preFrame method of all plugins
    void preFrame();
    //! call postFrame method of all plugins
    void postFrame();
    //! call preDraw method of all plugins
    void preDraw(osg::RenderInfo &renderInfo);
    //! call preSwapBuffers method of all plugins
    void preSwapBuffers(int windowNumber);
    //! call postSwapBuffers method of all plugins
    void postSwapBuffers(int windowNumber);
    //! call param method of all plugins
    void param(const char *paramName, bool inMapLoading);
    //! call init method of all plugins
    void init();
    //! call key method of all plugins
    bool key(int type, int keySym, int mod);
    //! call userEvent method of all plugins
    bool userEvent(int mod);
    //! call setTimestep method of all plugins
    void setTimestep(int timestep);
    //! send a message to all plugins
    void message(int t, int l, const void *b);
    //! add new plugins, if not already loaded
    //! unpack and distribute a Message
    void forwardMessage(int len, const void *buf);
    //! request to terminate COVER or COVISE session
    void requestQuit(bool killSession);
    //! send a message to COVISE/visualisation system - delivered via only one plugin
    bool sendVisMessage(const covise::Message *msg);
    //! request to become master of a collaborative session - return true if delivered
    bool becomeCollaborativeMaster();
    //! for visualisation system plugins: wait for message, return NULL if no such plugin
    virtual covise::Message *waitForVisMessage(int messageType);
    //! for visualisation system plugins: execute data flow network - return true if delivered
    virtual bool executeAll();
    //! allow plugins to expand scene bounding sphere
    void expandBoundingSphere(osg::BoundingSphere &bs);

private:
    coVRPluginList();
    ~coVRPluginList();
    //! unload all queued plugins
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
};
}
#endif
