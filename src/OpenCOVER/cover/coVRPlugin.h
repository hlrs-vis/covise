/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVR_PLUGIN_H
#define COVR_PLUGIN_H

/*! \file
 \brief  OpenCOVER plugin interface, derive plugins from coVRPlugin

 \author
 \author (C)
         HLRS, University of Stuttgart,
         Nobelstrasse 19,
         70569 Stuttgart,
         Germany

 \date
 */

#include <util/coExport.h>
#include <osg/Matrix>
#include <osg/Drawable>
#include <cover/coVRDynLib.h>
#include <cstdlib>
#include <OpenConfig/array.h>
#include <OpenConfig/value.h>
#include <OpenConfig/file.h>

#include <memory>
#include <string>

namespace grmsg
{
class coGRMsg;
}

namespace osg
{
class Node;
class Group;
}

namespace vrui
{
class coMenuItem;
}

namespace covise
{
class Message;
class UdpMessage;
class DataHandle;
}

// use COVERPLUGIN(YourMainPluginClass) in your plugin implementation
#define COVERPLUGIN(Plugin)                                         \
    extern "C" PLUGINEXPORT opencover::coVRPlugin *coVRPluginInit() \
    {                                                               \
        return new Plugin();                                        \
    }

namespace opencover
{
class RenderObject;
class coInteractor;
class Url;

//! currently, there are these methods to start a plugin:
//! 1. the plugin is specified in the configuration, then it is initialized in COVER before the render loop starts
//! 2. a plugin starts another plugin
//! 3. a Feedback object attached to a COVISE object is received
//! 4. upon user request from the TabletUI
//
//! make sure to clean up properly in the plugin's dtor


class COVEREXPORT coVRPlugin
{
    friend class coVRPluginList;

public:
    enum NotificationLevel {
        Info,
        Warning,
        Error,
        Fatal
    };

    //! called early, if loaded from config, before COVER is fully initialized
    coVRPlugin(const std::string &name);

    //! called before plugin is unloaded
    virtual ~coVRPlugin() = default;

    //! this function is called when COVER is up and running and the plugin is initialized
    virtual bool init()
    {
        return true;
    }
    
    //! this function is called when files have been loaded
    virtual bool init2()
    {
        return true;
    }

    //! reimplement to do early cleanup work and return false to prevent unloading
    virtual bool destroy()
    {
        return true;
    }

    //! retrieve the plugin's name
    const char *getName() const
    {
        return m_name.c_str();
    }


    std::shared_ptr<config::File> config();

    template<class V>
    std::unique_ptr<config::Value<V>> config(const std::string &section, const std::string &name, const V &defVal,
                                             config::Flag flags = config::Flag::Default);
    std::unique_ptr<config::Value<bool>> configBool(const std::string &section, const std::string &name,
                                                    const bool &defVal, config::Flag flags = config::Flag::Default);
    std::unique_ptr<config::Value<int64_t>> configInt(const std::string &section, const std::string &name,
                                                      const int64_t &defVal,
                                                      config::Flag flags = config::Flag::Default);
    std::unique_ptr<config::Value<double>> configFloat(const std::string &section, const std::string &name,
                                                       const double &defVal,
                                                       config::Flag flags = config::Flag::Default);
    std::unique_ptr<config::Value<std::string>> configString(const std::string &section, const std::string &name,
                                                             const std::string &defVal,
                                                             config::Flag flags = config::Flag::Default);

    template<class V>
    std::unique_ptr<config::Array<V>> configArray(const std::string &section, const std::string &name,
                                                  const std::vector<V> &defVal,
                                                  config::Flag flags = config::Flag::Default);
    std::unique_ptr<config::Array<bool>> configBoolArray(const std::string &section, const std::string &name,
                                                         const std::vector<bool> &defVal,
                                                         config::Flag flags = config::Flag::Default);
    std::unique_ptr<config::Array<int64_t>> configIntArray(const std::string &section, const std::string &name,
                                                           const std::vector<int64_t> &defVal,
                                                           config::Flag flags = config::Flag::Default);
    std::unique_ptr<config::Array<double>> configFloatArray(const std::string &section, const std::string &name,
                                                            const std::vector<double> &defVal,
                                                            config::Flag flags = config::Flag::Default);
    std::unique_ptr<config::Array<std::string>> configStringArray(const std::string &section, const std::string &name,
                                                                  const std::vector<std::string> &devVal,
                                                                  config::Flag flags = config::Flag::Default);

    //! this function is called when COVER wants to display a message to the user
    virtual void notify(NotificationLevel level, const char *text)
    {
        (void)level;
        (void)text;
    }

    /// first parameter is a pointer to the scene graph node,
    /// second parameter is a pointer to the COVISE object,
    /// NULL if not a COVISE object
    //! called when COVER adds a new COVISE object to the scenegraph or if other plugins insert a node into the scene graph
    virtual void addNode(osg::Node *, const RenderObject * = NULL)
    {
    }

    virtual void addNodeFromPlugin(osg::Node *node, const RenderObject *ro, coVRPlugin *addingPlugin)
    {
        return addNode(node, ro);
    }

    //! this function is called if a node in the scene graph is removed
    virtual void removeNode(osg::Node *, bool isGroup, osg::Node *realNode)
    {
        (void)isGroup;
    }

    //! this function is called whenever a COVISE object is received
    virtual void addObject(const RenderObject *container, osg::Group *parent,
            const RenderObject *geometry, const RenderObject *normals, const RenderObject *colors, const RenderObject *texture)
    {
        (void)container;
        (void)parent;
        (void)geometry;
        (void)normals;
        (void)colors;
        (void)texture;
    }

    //! this function is called when a COVISE Object is removed
    virtual void removeObject(const char *objName, bool replaceFlag)
    {
        (void)objName;
        (void)replaceFlag;
    }

    //! this function is called when COVER gets a new COVISE object with feedback attributes
    virtual void newInteractor(const RenderObject *container, coInteractor *it)
    {
        (void)container;
        (void)it;
    }

    //! this function is called when COVER wants to enable interaction with an interactor, return true if plugin accepts request
    virtual bool requestInteraction(coInteractor *inter, osg::Node *triggerNode, bool isMouse)
    {
        (void)inter;
        (void)triggerNode;
        (void)isMouse;
        return false;
    }

    //! this function is called if a error message from the controller is received
    virtual void coviseError(const char *error)
    {
        (void)error;
    }

    //! this function is called if a message from the gui is received
    virtual void guiToRenderMsg(const grmsg::coGRMsg &msg) 
    {
        (void)(msg);
    };

    //! this function is called from the main thread after the state for a frame is set up, just before preFrame()
    //! return true, if you need the scene to be rendered immediately
    virtual bool update()
    {
        return false; // don't request that scene be re-rendered
    }

    //! this function is called from the main thread before rendering a frame
    virtual void preFrame()
    {
    }

    //! this function is called from the main thread after a frame was rendered
    virtual void postFrame()
    {
    }

    //! this function is called from the draw thread before drawing the scenegraph (after drawing the AR background)
    virtual void preDraw(osg::RenderInfo &)
    {
    }

    //! this function is called from the draw thread before swapbuffers
    virtual void preSwapBuffers(int /*windowNumber*/)
    {
    }

    //! this function is called from the main thread after rendering has finished on all nodes and before any swap buffers happen
    virtual void clusterSyncDraw()
    {
    }

    //! this function is called from the draw thread after swapbuffers
    virtual void postSwapBuffers(int /*windowNumber*/)
    {
    }

    //! this function is called whenever a module parameter of the renderer has changed
    virtual void param(const char *paramName, bool inMapLoading)
    {
        (void)paramName;
        (void)inMapLoading;
    }

    //! this functions is called when a key is pressed or released
    virtual void key(int type, int keySym, int mod)
    {
        (void)type;
        (void)keySym;
        (void)mod;
    }

    //! this functions is called when a user event arrives
    virtual void userEvent(int mod)
    {
        (void)mod;
    }

    // this function is called if a message arrives
    virtual void message(int toWhom, int type, int length, const void *data)
    {
        (void)toWhom;
        (void)type;
        (void)length;
        (void)data;
    }
	// this function is called if a UDPmessage arrives
	virtual void UDPmessage(covise::UdpMessage* msg)
	{
		(void)msg;
	}


    //! this functions is called when the current timestep is changed
    //! plugins should display this timestep if possible
    //! if there are not enough timesteps available, display the last timestep
    virtual void setTimestep(int t)
    {
        (void)t;
    }

    //! the plugin should prepare to display timestep t
    //! and call commitTimestep(t) when ready
    virtual void requestTimestep(int t)
    {
        commitTimestep(t);
    }

    //! for Trackingsystem plugins: return the Matrix of device station
    virtual void getMatrix(int station, osg::Matrix &mat)
    {
        (void)station;
        (void)mat;
    }

    //! for Trackingsystem plugins: return the button state of device station
    virtual unsigned int button(int station)
    {
        (void)station;
        return 0;
    }

    //! for Trackingsystem plugins: return the wheel state of device station
    virtual int wheel(int station)
    {
        (void)station;
        return 0;
    }

    //! for visualisation system plugins: request to terminate COVER or COVISE session
    virtual void requestQuit(bool killSession)
    {
        (void)killSession;
    }

    //! for visualisation system plugins: send message to system - return true if delivered
    virtual bool sendVisMessage(const covise::Message *msg)
    {
        (void)msg;
        return false;
    }

    //! for visualisation system plugins: request to become master - return true if delivered
    virtual bool becomeCollaborativeMaster()
    {
        return false;
    }

    //! for visualisation system plugins: return string identifying collaborative session uniquely
    virtual std::string collaborativeSessionId() const
    {
        return "";
    }


    //! for visualisation system plugins: wait for message, return NULL if no such plugin
    virtual covise::Message *waitForVisMessage(int messageType)
    {
        (void)messageType;
        return NULL;
    }

    //! for visualisation system plugins: execute data flow network - return true if delivered
    virtual bool executeAll()
    {
        return false;
    }

    //! allow plugin to expand bounding sphere
    virtual void expandBoundingSphere(osg::BoundingSphere &bs)
    {
        (void)bs;
    }

    virtual bool windowCreate(int num)
    {
        (void)num;
        return false;
    }

    virtual void windowCheckEvents(int num)
    {
        (void)num;
    }

    virtual void windowUpdateContents(int num)
    {
        (void)num;
    }

    virtual void windowDestroy(int num)
    {
        (void)num;
    }

    virtual void windowFullScreen(int num, bool state)
    {
        (void)num;
        (void)state;
    }

    //! let the plugin that has a grab on viewer update viewer matrix
    virtual bool updateViewer()
    {
        return false; // no update, no re-render required
    }

protected:
    //! call as a response to requestTimestep(t) when timestep t is prepared
    void commitTimestep(int t);

private:
    void requestTimestepWrapper(int t);

    bool m_initDone = false;
    const std::string m_name;
    CO_SHLIB_HANDLE handle = nullptr;
    int m_outstandingTimestep = -1;
    std::shared_ptr<config::File> m_configFile;
};
}
#endif
