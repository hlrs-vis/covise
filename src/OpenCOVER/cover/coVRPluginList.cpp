/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>
#include <util/byteswap.h>
#include <util/string_util.h>

#include "coVRPluginList.h"
#include "coVRPluginSupport.h"
#include "coVRPlugin.h"
#include "coVRSelectionManager.h"
#include "RenderObject.h"
#include "coVRMSController.h"
#include "coVRAnimationManager.h"

#include <config/CoviseConfig.h>
#include "coVRTui.h"
#include "OpenCOVER.h"
#include "coHud.h"

#ifdef DOTIMING
#include <util/coTimer.h>
#endif

#include "VRViewer.h"
#include "PluginMenu.h"
#include "coVRConfig.h"

namespace opencover
{
class coInteractor;
}

using namespace opencover;
using namespace covise;

typedef opencover::coVRPlugin *(coVRPluginInitFunc)();

// do something for all plugins
#define DOALL(something)                                                                                   \
    {                                                                                                      \
        for (PluginMap::const_iterator plugin_it = m_plugins.begin(); plugin_it != m_plugins.end(); ++plugin_it) \
        {                                                                                                  \
            coVRPlugin *plugin = plugin_it->second;                                                        \
            {                                                                                              \
                something;                                                                                 \
            }                                                                                              \
        }                                                                                                  \
    }

coVRPlugin *coVRPluginList::loadPlugin(const char *name, bool showErrors)
{
    if (cover->debugLevel(3))
    {
        if (name)
            fprintf(stderr, "coVRPluginList::loadPlugin %s\n", name);
        else
            fprintf(stderr, "coVRPluginList::loadPlugin name=NULL\n");
    }

    std::string libName = coVRDynLib::libName(name);
    CO_SHLIB_HANDLE handle = coVRDynLib::dlopen(libName, showErrors);
    if (handle == NULL)
    {
        if (showErrors)
            cerr << "ERROR: could not load shared Library " << libName << endl;
        return NULL;
    }
    coVRPluginInitFunc *initFunc = (coVRPluginInitFunc *)coVRDynLib::dlsym(handle, "coVRPluginInit");
    if (initFunc == NULL)
    {
        coVRDynLib::dlclose(handle);
        if (showErrors)
            cerr << "ERROR: malformed COVER plugin " << name << ", no coVRPluginInit defined" << endl;
        return NULL;
    }
    coVRPlugin *plugin = initFunc();
    if (!plugin)
    {
        coVRDynLib::dlclose(handle);
        if (showErrors)
            cerr << "ERROR: in COVER plugin " << name << ", coVRPluginInit failed" << endl;
        return NULL;
    }

    plugin->handle = handle;
    plugin->setName(name);

    return plugin;
}

coVRPluginList *coVRPluginList::singleton = NULL;
coVRPluginList *coVRPluginList::instance()
{
    if (!singleton)
        singleton = new coVRPluginList();
    return singleton;
}

coVRPluginList::~coVRPluginList()
{
    for (int d=0; d<NumPluginDomains; ++d)
        unloadAllPlugins(static_cast<PluginDomain>(d));
    singleton = NULL;
}

void coVRPluginList::unloadAllPlugins(PluginDomain domain)
{
    if (domain == Window)
        return;

    bool wasThreading = false;
    bool havePlugins = !m_loadedPlugins[domain].empty();

    if (havePlugins)
    {
        if (cover->debugLevel(1))
            cerr << "Unloading plugins (domain " << domain << "):";

        if (domain == Default)
            wasThreading = VRViewer::instance()->areThreadsRunning();
        if (wasThreading)
            VRViewer::instance()->stopThreading();
    }

    while (!m_loadedPlugins[domain].empty())
    {
        coVRPlugin *plug = m_loadedPlugins[domain].back();
        if (plug)
        {
            if (cover->debugLevel(1))
                cerr << " " << plug->getName();
            m_unloadQueue.push_back(plug->handle);
            plug->destroy();
        }
        unmanage(plug);
        delete plug;
    }
    unloadQueued();

    if (havePlugins)
    {
        if (wasThreading)
            VRViewer::instance()->startThreading();
        if (cover->debugLevel(1))
            cerr << endl;
    }
}

coVRPluginList::coVRPluginList()
{
    singleton = this;
    m_requestedTimestep = -1;
    m_numOutstandingTimestepPlugins = 0;
    keyboardPlugin = NULL;
}

void coVRPluginList::loadDefault()
{
    if (cover->debugLevel(1))
    {
        cerr << "Loading plugins:";
    }

    std::vector<std::string> plugins;
    if (const char *env = getenv("COVER_PLUGINS"))
    {
        plugins = split(env, ':');
    }

    coCoviseConfig::ScopeEntries pluginNameEntries = coCoviseConfig::getScopeEntries("COVER.Plugin");
    const char **pluginNames = pluginNameEntries.getValue();
    if (pluginNames != NULL)
    {
        int i = 0;
        while (pluginNames[i])
        {
            char *entryName = new char[strlen(pluginNames[i]) + 100];
            sprintf(entryName, "COVER.Plugin.%s", pluginNames[i]);
            if (coCoviseConfig::isOn(entryName, false))
            {
                plugins.push_back(pluginNames[i]);
            }

            bool exists = false;
            if (coCoviseConfig::isOn("menu", entryName, false, &exists))
            {
                coVRPlugin *m = getPlugin(pluginNames[i]);
                PluginMenu::instance()->addEntry(pluginNames[i], m);
            }

            delete[] entryName;
            i++; // skip name
            i++; //skip value (may be NULL)
        }
    }
    if(!coVRConfig::instance()->viewpointsFile.empty())
    {
        plugins.push_back("ViewPoint");
    }

    std::vector<std::string> failed;
    for (size_t i = 0; i < plugins.size(); ++i)
    {
        if (cover->debugLevel(1))
        {
            cerr << " " << plugins[i];
        }
        if (!getPlugin(plugins[i].c_str()))
        {
            if (coVRPlugin *m = loadPlugin(plugins[i].c_str()))
            {
                manage(m, Default); // if init OK, then add new plugin
            }
            else
            {
                failed.push_back(plugins[i]);
            }
        }
    }
    if (cover->debugLevel(1))
    {
        cerr << "." << endl;
    }

    if (!failed.empty())
    {
        cerr << "Plugins which failed to load:";
        for(size_t i=0; i<failed.size(); ++i)
        {
            cerr << " " << failed[i];
        }
        cerr << "." << endl;
    }
}

void coVRPluginList::unload(coVRPlugin *plugin)
{
    if (!plugin)
        return;

    bool wasThreading = VRViewer::instance()->areThreadsRunning();
    if (wasThreading)
        VRViewer::instance()->stopThreading();
    if (plugin->destroy())
    {
        m_unloadQueue.push_back(plugin->handle);
        unmanage(plugin);
        delete plugin;
        updateState();
    }
    if (wasThreading)
        VRViewer::instance()->startThreading();
}

void coVRPluginList::unloadQueued()
{
    for (HandleList::iterator it = m_unloadNext.begin(); it != m_unloadNext.end(); ++it)
    {
        coVRDynLib::dlclose(*it);
    }
    m_unloadNext = m_unloadQueue;
    m_unloadQueue.clear();
}

void coVRPluginList::manage(coVRPlugin *plugin, PluginDomain domain)
{
    m_plugins[plugin->getName()] = plugin;
    m_loadedPlugins[domain].push_back(plugin);
}

void coVRPluginList::unmanage(coVRPlugin *plugin)
{
    if (!plugin)
        return;

    auto it = m_plugins.find(plugin->getName());
    if (it == m_plugins.end())
    {
        cerr << "Plugin to unload not found2: " << plugin->getName() << endl;
    }
    else
    {
        m_plugins.erase(it);
    }

    for (int d=0; d<NumPluginDomains; ++d)
    {
        auto it2 = std::find(m_loadedPlugins[d].begin(), m_loadedPlugins[d].end(), plugin);
        if (it2 != m_loadedPlugins[d].end())
        {
            m_loadedPlugins[d].erase(it2);
        }
    }
}

void coVRPluginList::notify(int level, const char *text) const
{
    DOALL(plugin->notify((coVRPlugin::NotificationLevel)level, text));
}

void coVRPluginList::addNode(osg::Node *node, const RenderObject *ro, coVRPlugin *addingPlugin) const
{
    DOALL(if (plugin != addingPlugin)
              plugin->addNode(node, const_cast<RenderObject *>(ro)));
}

void coVRPluginList::addObject(const RenderObject *container, osg::Group *parent,
        const RenderObject *geometry, const RenderObject *normals, const RenderObject *colors, const RenderObject *texture) const
{
    // call addObject for the current plugin in the plugin list
    DOALL(plugin->addObject(container, parent, geometry, normals, colors, texture));
}

void coVRPluginList::newInteractor(const RenderObject *container, coInteractor *it) const
{
    DOALL(plugin->newInteractor(container, it));
}

bool coVRPluginList::requestInteraction(coInteractor *inter, osg::Node *triggerNode, bool isMouse)
{
    DOALL(if (plugin->requestInteraction(inter, triggerNode, isMouse))
          return true);
    return false;
}

void coVRPluginList::coviseError(const char *error) const
{
    DOALL(plugin->coviseError(error));
}

void coVRPluginList::guiToRenderMsg(const char *msg) const
{
    DOALL(plugin->guiToRenderMsg(msg));
}

void coVRPluginList::removeObject(const char *objName, bool replaceFlag) const
{
    // call deleteObject for the current plugin in the plugin list
    DOALL(plugin->removeObject(objName, replaceFlag));
}

void coVRPluginList::removeNode(osg::Node *node, bool isGroup, osg::Node *realNode) const
{
    if (isGroup)
        coVRSelectionManager::instance()->removeNode(node);

    DOALL(plugin->removeNode(node, isGroup, realNode));
}

bool coVRPluginList::update() const
{
    bool ret = false;
#ifdef DOTIMING
    MARK0("COVER calling update for all plugins");
#endif
    DOALL(ret |= plugin->update());
#ifdef DOTIMING
    MARK0("done");
#endif
    return ret;
}

void coVRPluginList::preFrame()
{
#ifdef DOTIMING
    MARK0("COVER calling preFrame for all plugins");
#endif
    unloadQueued();

    DOALL(plugin->preFrame());
#ifdef DOTIMING
    MARK0("done");
#endif
}

void coVRPluginList::setTimestep(int t) const
{
    DOALL(plugin->setTimestep(t));
}

void coVRPluginList::requestTimestep(int t)
{
    assert(m_requestedTimestep == -1);
    m_requestedTimestep = t;
    assert(m_numOutstandingTimestepPlugins == 0);
    ++m_numOutstandingTimestepPlugins;
    DOALL(++m_numOutstandingTimestepPlugins; plugin->requestTimestepWrapper(t));
    commitTimestep(t, NULL);
}

void coVRPluginList::commitTimestep(int t, coVRPlugin *plugin)
{
    assert(m_requestedTimestep == t);
    assert(m_numOutstandingTimestepPlugins > 0);
    --m_numOutstandingTimestepPlugins;
    if (m_numOutstandingTimestepPlugins < 0)
    {
        std::cerr << "coVRPluginList: plugin " << plugin->getName() << " overcommitted timestep " << t << std::endl;
        m_numOutstandingTimestepPlugins = 0;
    }
    if (m_numOutstandingTimestepPlugins <= 0) {
        coVRAnimationManager::instance()->setAnimationFrame(t);
        m_requestedTimestep = -1;
    }
}

void coVRPluginList::postFrame() const
{
#ifdef DOTIMING
    MARK0("COVER calling postFrame for all plugins");
#endif

    DOALL(plugin->postFrame());
#ifdef DOTIMING
    MARK0("done");
#endif
}

void coVRPluginList::preDraw(osg::RenderInfo &renderInfo) const
{
    DOALL(plugin->preDraw(renderInfo));
}

void coVRPluginList::preSwapBuffers(int windowNumber) const
{
    DOALL(plugin->preSwapBuffers(windowNumber));
}

void coVRPluginList::clusterSyncDraw() const
{
    DOALL(plugin->clusterSyncDraw());
}

void coVRPluginList::postSwapBuffers(int windowNumber) const
{
    DOALL(plugin->postSwapBuffers(windowNumber));
}

void coVRPluginList::param(const char *paramName, bool inMapLoading) const
{
    DOALL(plugin->param(paramName, inMapLoading));
}

void coVRPluginList::grabKeyboard(coVRPlugin *p)
{
    keyboardPlugin = p;
}

bool coVRPluginList::key(int type, int keySym, int mod) const
{
    if (keyboardPlugin)
    {
        keyboardPlugin->key(type, keySym, mod); // only forward key events to this plugin
    }
    else
    {
        DOALL(plugin->key(type, keySym, mod));
    }

    return true;
}

bool coVRPluginList::userEvent(int mod) const
{
    DOALL(plugin->userEvent(mod));
    return true;
}

void coVRPluginList::init()
{
    PluginMenu::instance()->init();
    for (PluginMap::iterator it = m_plugins.begin();
         it != m_plugins.end();)
    {
        OpenCOVER::instance()->hud->setText3(it->first);
        OpenCOVER::instance()->hud->redraw();
        auto plug = it->second;
        ++it;

        if (!plug->m_initDone && !plug->init())
        {
            cerr << "plugin " << plug->getName() << " failed to initialise" << endl;
            unmanage(plug);
        }
        else
        {
            plug->m_initDone = true;
        }
    }
    updateState();
}
void coVRPluginList::init2()
{
    DOALL(plugin->init2());
}

void coVRPluginList::message(int toWhom, int t, int l, const void *b) const
{
    DOALL(plugin->message(toWhom, t, l, b));
}

coVRPlugin *coVRPluginList::getPlugin(const char *name) const
{
    PluginMap::const_iterator it = m_plugins.find(name);
    if (it == m_plugins.end())
        return NULL;

    return it->second;
}

coVRPlugin *coVRPluginList::addPlugin(const char *name, PluginDomain domain)
{
    std::string arg(name);
    std::string n = coVRMSController::instance()->syncString(arg);
    if (n != arg)
    {
        std::cerr << "coVRPluginList::addPlugin(" << arg << "), but master is trying to load " << n << std::endl;
        abort();
    }
    coVRPlugin *m = getPlugin(name);
    if (m == NULL)
    {
        m = loadPlugin(name, domain == Default);
        if (m && m->init())
        {
            manage(m, domain);
            m->m_initDone = true;
            m->init2();
        }
        else
        {
            if (domain == Default)
                cerr << "plugin " << name << " failed to initialise" << endl;
            delete m;
            m = NULL;
        }
        if (domain == Default)
            updateState();
    }
    return m;
}

void coVRPluginList::forwardMessage(int len, const void *buf) const
{
    int headerSize = 2 * sizeof(int);
    if (len < headerSize)
    {
        cerr << "wrong Message received in coVRPluginList::forwardMessage" << endl;
        return;
    }
    int toWhom = *((int *)buf);
    int type = *(((int *)buf) + 1);
#ifdef BYTESWAP
    byteSwap(toWhom);
    byteSwap(type);
#endif
    if ((toWhom == coVRPluginSupport::TO_ALL)
        || (toWhom == coVRPluginSupport::TO_ALL_OTHERS)
        || (toWhom == coVRPluginSupport::VRML_EVENT))
    {
        message(toWhom, type, len - headerSize, ((const char *)buf) + headerSize);
    }
    else
    {
        const char *name = (((const char *)buf) + headerSize);
        coVRPlugin *mod = getPlugin(name);
        if (mod)
        {
            int ssize = strlen(name) + 1 + (8 - ((strlen(name) + 1) % 8));
            mod->message(toWhom, type, len - headerSize - ssize, ((const char *)buf) + headerSize + ssize);
        }
    }
}

void coVRPluginList::requestQuit(bool killSession) const
{
    DOALL(plugin->requestQuit(killSession));
}

bool coVRPluginList::sendVisMessage(const Message *msg) const
{
    DOALL(if (plugin->sendVisMessage(msg))
          { return true;
          })
    return false;
}

bool coVRPluginList::becomeCollaborativeMaster() const
{
    DOALL(if (plugin->becomeCollaborativeMaster())
          { return true;
          })

    return false;
}

covise::Message *coVRPluginList::waitForVisMessage(int messageType) const
{
    DOALL(covise::Message *m = plugin->waitForVisMessage(messageType);
          if (m)
          { return m;
    });

    return NULL;
}

bool coVRPluginList::executeAll() const
{
    DOALL(if (plugin->executeAll())
          { return true;
          });

    return false;
}

void coVRPluginList::expandBoundingSphere(osg::BoundingSphere &bs) const
{
    DOALL(plugin->expandBoundingSphere(bs));
}

void coVRPluginList::updateState()
{
    coVRTui::instance()->updateState();
    PluginMenu::instance()->updateState();
}
