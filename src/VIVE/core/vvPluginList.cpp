/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/common.h>
#include <util/byteswap.h>
#include <util/string_util.h>

#include "vvPluginList.h"
#include "vvPluginSupport.h"
#include "vvPlugin.h"
//#include "vvSelectionManager.h"
//#include "vvRenderObject.h"
#include "vvMSController.h"
//#include "vvAnimationManager.h"

#include <config/CoviseConfig.h>
//#include "vvTui.h"
#include "vvVIVE.h"
//#include "vvHud.h"

#ifdef DOTIMING
#include <util/coTimer.h>
#endif

//#include "vvPluginMenu.h"
#include "vvConfig.h"
#include <vsg/app/Viewer.h>

namespace vive
{
class vvInteractor;
}

using namespace vive;
using namespace covise;

typedef vive::vvPlugin *(vvPluginInitFunc)();

// do something for all plugins
#define DOALL(something) \
    { \
        m_stopIteration = false; \
        for (PluginMap::const_iterator plugin_it = m_plugins.begin(); plugin_it != m_plugins.end(); ++plugin_it) \
        { \
            vvPlugin *plugin = plugin_it->second; \
            { \
                something; \
            } \
            if (m_stopIteration) \
                break; \
        } \
    }

vvPlugin *vvPluginList::loadPlugin(const char *name, bool showErrors)
{
    m_stopIteration = true;

    std::string libName = vvDynLib::libName(name);
    CO_SHLIB_HANDLE handle = vvDynLib::dlopen(libName, showErrors);
    if (handle == NULL)
    {
        if (showErrors)
        {
            cerr << "ERROR: could not load shared Library " << libName << endl;
            auto showenv = [](const std::string &var)
            {
                const char *val = getenv(var.c_str());
                if (val)
                {
                    std::cerr << var << "=" << val << std::endl;
                }
                else
                {
                    std::cerr << var << " not set" << std::endl;
                }
            };
#if defined(__APPLE__)
            showenv("VISTLE_DYLD_LIBRARY_PATH");
            showenv("DYLD_LIBRARY_PATH");
            showenv("DYLD_FRAMEWORK_PATH");
            showenv("DYLD_FALLBACK_LIBRARY_PATH");
            showenv("DYLD_FALLBACK_FRAMEWORK_PATH");
#elif defined(__linux)
            showenv("LD_LIBRARY_PATH");
#else
#endif
            showenv("PATH");
        }
        return NULL;
    }
    vvPluginInitFunc *initFunc = (vvPluginInitFunc *)vvDynLib::dlsym(handle, "vvPluginInit");
    if (initFunc == NULL)
    {
        vvDynLib::dlclose(handle);
        if (showErrors)
            cerr << "ERROR: malformed VIVE plugin " << name << ", no vvPluginInit defined" << endl;
        return NULL;
    }
    vvPlugin *plugin = initFunc();
    if (!plugin)
    {
        vvDynLib::dlclose(handle);
        if (showErrors)
            cerr << "ERROR: in VIVE plugin " << name << ", vvPluginInit failed" << endl;
        return NULL;
    }

    plugin->handle = handle;
    return plugin;
}

vvPluginList *vvPluginList::singleton = NULL;
vvPluginList *vvPluginList::instance()
{
    if (!singleton)
        singleton = new vvPluginList();
    return singleton;
}

vvPluginList::~vvPluginList()
{
    for (int d=0; d<NumPluginDomains; ++d)
        unloadAllPlugins(static_cast<PluginDomain>(d));
  //  delete vvPluginMenu::instance();
    singleton = NULL;
}

void vvPluginList::unloadAllPlugins(PluginDomain domain)
{
#if 0
    if (domain == Window)
        return;
#endif

    bool wasThreading = false;
    bool havePlugins = !m_loadedPlugins[domain].empty();

    if (havePlugins)
    {
        /*
        if (domain == Default)
            wasThreading = vvViewer::instance()->areThreadsRunning();
        if (wasThreading)
            vvViewer::instance()->stopThreading();*/
    }

    while (!m_loadedPlugins[domain].empty())
    {
        vvPlugin *plug = m_loadedPlugins[domain].back();
        if (plug)
        {
            m_unloadQueue.push_back(plug->handle);
            plug->destroy();

            vvPluginSupport::instance()->preparePluginUnload();
        }
        unmanage(plug);
        delete plug;
    }

    if (havePlugins)
    {
   //     if (wasThreading)
    //        vvViewer::instance()->startThreading();
        if (vvPluginSupport::instance()->debugLevel(1))
            cerr << endl;
    }
}

std::vector<std::string> getSharedPlugins()
{
    std::vector<std::string> sharedPlugins;
    coCoviseConfig::ScopeEntries pluginNames = coCoviseConfig::getScopeEntries("VIVE.Plugin");
    for (const auto &pluginName : pluginNames)
    {
        if (coCoviseConfig::isOn("shared", std::string("VIVE.Plugin.") + pluginName.first, false))
        {
            sharedPlugins.push_back(pluginName.first);
        }
    }
    return sharedPlugins;
}

vvPluginList::vvPluginList()
{
    singleton = this;
    m_requestedTimestep = -1;
    m_numOutstandingTimestepPlugins = 0;
    keyboardPlugin = NULL;
    std::vector<std::string> sharedPlugins = getSharedPlugins();

    for(const auto& plugin : sharedPlugins)
    {
        auto p = m_sharedPlugins.emplace(std::make_pair(plugin, std::unique_ptr < vrb::SharedState<bool>> {new vrb::SharedState<bool>{"vvPluginList_" + plugin}}));
        p.first->second->setUpdateFunction([p, plugin, this]() {
            if (p.first->second->value())
            {
                addPlugin(p.first->first.c_str());
            }
            else
            {
                auto *m = getPlugin(p.first->first.c_str());
                if (m)
                {
                    unload(m);
                }
            }
        });

    }
}

void vvPluginList::loadDefault()
{
    if (vvPluginSupport::instance()->debugLevel(1))
    {
        cerr << "Loading plugins:";
    }

    std::vector<std::string> plugins;
    std::vector<bool> addToMenu;
    if (const char *env = getenv("VIVE_PLUGINS"))
    {
        plugins = split(env, ':');
        addToMenu.push_back(true);
    }

    coCoviseConfig::ScopeEntries pluginNames = coCoviseConfig::getScopeEntries("VIVE.Plugin");
    for (const auto &pluginName : pluginNames)
    {
        std::string entryName = "VIVE.Plugin." + pluginName.first;
        if (coCoviseConfig::isOn("menu", entryName, false))
        {
            vvPlugin *m = getPlugin(pluginName.first.c_str());
         //   vvPluginMenu::instance()->addEntry(pluginName.first, m);
            auto it = std::find(plugins.begin(), plugins.end(), std::string(entryName));
            if (it != plugins.end()) {
                addToMenu[it-plugins.begin()] = false;
            }
        }

        if (coCoviseConfig::isOn(entryName, false))
        {
            plugins.push_back(pluginName.first);
        }
    }
    if (!vvConfig::instance()->viewpointsFile.empty())
    {
        plugins.push_back("ViewPoint");
    }

    for (size_t i=0; i<addToMenu.size(); ++i)
    {
        if (addToMenu[i])
        {
            vvPlugin *m = getPlugin(plugins[i].c_str());
 //           vvPluginMenu::instance()->addEntry(plugins[i], m);
        }
    }

    auto browserDefaultUrl = getenv("COVISE_BROWSER_INIT_URL");
    if(browserDefaultUrl)
        plugins.push_back("Browser");
        
    std::vector<std::string> failed;
    for (size_t i = 0; i < plugins.size(); ++i)
    {
        if (vvPluginSupport::instance()->debugLevel(1))
        {
            cerr << " " << plugins[i];
        }
        if (!getPlugin(plugins[i].c_str()))
        {
            if (vvPlugin *m = loadPlugin(plugins[i].c_str()))
            {
                manage(m, Default); // if init OK, then add new plugin
            }
            else
            {
                failed.push_back(plugins[i]);
            }
        }
    }
    if (vvPluginSupport::instance()->debugLevel(1))
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

void vvPluginList::unload(vvPlugin *plugin)
{
    if (!plugin)
        return;

  //  bool wasThreading = vvViewer::instance()->areThreadsRunning();
  //  if (wasThreading)
  //      vvViewer::instance()->stopThreading();
    if (plugin->destroy())
    {
        m_unloadQueue.push_back(plugin->handle);
		vvPluginSupport::instance()->preparePluginUnload();
        unmanage(plugin);
        auto shared = m_sharedPlugins.find(plugin->getName());
        if (shared != m_sharedPlugins.end())
        {
            *shared->second = false;
        }
        delete plugin;
        updateState();
    }
  //  if (wasThreading)
   //     vvViewer::instance()->startThreading();
}

void vvPluginList::unloadQueued()
{
    for (HandleList::iterator it = m_unloadNext.begin(); it != m_unloadNext.end(); ++it)
    {
        vvDynLib::dlclose(*it);
    }
    m_unloadNext = m_unloadQueue;
    m_unloadQueue.clear();
}

void vvPluginList::manage(vvPlugin *plugin, PluginDomain domain)
{
    m_plugins[plugin->getName()] = plugin;
    m_loadedPlugins[domain].push_back(plugin);
}

void vvPluginList::unmanage(vvPlugin *plugin)
{
    if (!plugin)
        return;

    m_stopIteration = true;

    if (plugin == viewerPlugin)
    {
        std::cerr << "Plugin " << plugin->getName() << " did not release viewer grab before unloading" << std::endl;
        viewerPlugin = nullptr;
    }
    if (plugin == keyboardPlugin)
    {
        std::cerr << "Plugin " << plugin->getName() << " did not release keyboard grab before unloading" << std::endl;
        keyboardPlugin = nullptr;
    }

    auto it = m_plugins.find(plugin->getName());
    if (it == m_plugins.end())
    {
        cerr << "Plugin to unload not found2: " << plugin->getName() << endl;
    }
    else
    {
        m_plugins.erase(it);
        m_stopIteration = true;
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
/*
void vvPluginList::notify(int level, const char *text) const
{
    DOALL(plugin->notify((vvPlugin::NotificationLevel)level, text));
}*/

void vvPluginList::addNode(vsg::Node *node, const vvRenderObject *ro, vvPlugin *addingPlugin) const
{
    DOALL(if (plugin != addingPlugin) plugin->addNodeFromPlugin(node, const_cast<vvRenderObject *>(ro), addingPlugin));
}

void vvPluginList::addObject(const vvRenderObject *container, vsg::Group *parent,
        const vvRenderObject *geometry, const vvRenderObject *normals, const vvRenderObject *colors, const vvRenderObject *texture) const
{
    // call addObject for the current plugin in the plugin list
    DOALL(plugin->addObject(container, parent, geometry, normals, colors, texture));
}

void vvPluginList::newInteractor(const vvRenderObject *container, vvInteractor *it) const
{
    DOALL(plugin->newInteractor(container, it));
}

bool vvPluginList::requestInteraction(vvInteractor *inter, vsg::Node *triggerNode, bool isMouse)
{
    DOALL(if (plugin->requestInteraction(inter, triggerNode, isMouse))
          return true);
    return false;
}

void vvPluginList::coviseError(const char *error) const
{
    DOALL(plugin->coviseError(error));
}

void vvPluginList::guiToRenderMsg(const grmsg::coGRMsg &msg)  const
{
    DOALL(plugin->guiToRenderMsg(msg));
}

void vvPluginList::removeObject(const char *objName, bool replaceFlag) const
{
    // call deleteObject for the current plugin in the plugin list
    DOALL(plugin->removeObject(objName, replaceFlag));
}

void vvPluginList::removeNode(vsg::Node *node, bool isGroup, vsg::Node *realNode) const
{
 //   if (isGroup)
  //      vvSelectionManager::instance()->removeNode(node);

    DOALL(plugin->removeNode(node, isGroup, realNode));
}

bool vvPluginList::update() const
{
    bool ret = false;
#ifdef DOTIMING
    MARK0("VIVE calling update for all plugins");
#endif
    DOALL(ret |= plugin->update());
#ifdef DOTIMING
    MARK0("done");
#endif
    return ret;
}

void vvPluginList::preFrame()
{
#ifdef DOTIMING
    MARK0("VIVE calling preFrame for all plugins");
#endif
    unloadQueued();

    DOALL(plugin->preFrame());
#ifdef DOTIMING
    MARK0("done");
#endif
}

void vvPluginList::setTimestep(int t)
{
    m_currentTimestep = t;
    DOALL(plugin->setTimestep(t));
}

void vvPluginList::requestTimestep(int t)
{
    assert(m_requestedTimestep == -1);
    m_requestedTimestep = t;
    assert(m_numOutstandingTimestepPlugins == 0);
    ++m_numOutstandingTimestepPlugins;
    DOALL(++m_numOutstandingTimestepPlugins; plugin->requestTimestepWrapper(t));
    commitTimestep(t, NULL);
}

void vvPluginList::commitTimestep(int t, vvPlugin *plugin)
{
    if (t != m_requestedTimestep)
    {
        std::cerr << "vvPluginList: plugin " << plugin->getName() << " committed timestep " << t << ", but "
                  << m_requestedTimestep << " was expected" << std::endl;
    }
    assert(m_numOutstandingTimestepPlugins > 0);
    --m_numOutstandingTimestepPlugins;
    if (m_numOutstandingTimestepPlugins < 0)
    {
        std::cerr << "vvPluginList: plugin " << plugin->getName() << " overcommitted timestep " << t << std::endl;
        m_numOutstandingTimestepPlugins = 0;
    }
    if (m_numOutstandingTimestepPlugins <= 0) {
 //       vvAnimationManager::instance()->setAnimationFrame(t);
        m_requestedTimestep = -1;
    }
}

void vvPluginList::postFrame() const
{
#ifdef DOTIMING
    MARK0("VIVE calling postFrame for all plugins");
#endif

    DOALL(plugin->postFrame());
#ifdef DOTIMING
    MARK0("done");
#endif
}

void vvPluginList::preDraw() const
{
    DOALL(plugin->preDraw());
}

void vvPluginList::preSwapBuffers(int windowNumber) const
{
    DOALL(plugin->preSwapBuffers(windowNumber));
}

void vvPluginList::clusterSyncDraw() const
{
    DOALL(plugin->clusterSyncDraw());
}

void vvPluginList::postSwapBuffers(int windowNumber) const
{
    DOALL(plugin->postSwapBuffers(windowNumber));
}

void vvPluginList::param(const char *paramName, bool inMapLoading) const
{
    DOALL(plugin->param(paramName, inMapLoading));
}

void vvPluginList::grabKeyboard(vvPlugin *p)
{
    keyboardPlugin = p;
}

void vvPluginList::grabViewer(vvPlugin *p)
{
    viewerPlugin = p;
}

vvPlugin *vvPluginList::viewerGrabber() const
{
    return viewerPlugin;
}

bool vvPluginList::key(int type, int keySym, int mod) const
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

bool vvPluginList::userEvent(int mod) const
{
    DOALL(plugin->userEvent(mod));
    return true;
}

void vvPluginList::init()
{
    std::vector<vvPlugin *> failed;

  //  vvPluginMenu::instance()->init();
    for (PluginMap::iterator it = m_plugins.begin();
         it != m_plugins.end();)
    {
  //      vvVIVE::instance()->hud->setText3(it->first);
  //      vvVIVE::instance()->hud->redraw();
        auto plug = it->second;
        ++it;

        if (!plug->m_initDone && !plug->init())
        {
            cerr << "plugin " << plug->getName() << " failed to initialise" << endl;
            failed.push_back(plug);
        }
        else
        {
            plug->m_initDone = true;
        }
    }

    for (auto &plug: failed)
    {
        unmanage(plug);
    }

    updateState();
}
void vvPluginList::init2()
{
    DOALL(plugin->init2());
    DOALL(plugin->setTimestep(m_currentTimestep));
}

void vvPluginList::message(int toWhom, int t, int l, const void *b, const vvPlugin *exclude) const
{
    DOALL(if (plugin != exclude) plugin->message(toWhom, t, l, b));
}

void vvPluginList::UDPmessage(covise::UdpMessage* msg) const
{
	DOALL(plugin->UDPmessage(msg));
}

vvPlugin *vvPluginList::getPlugin(const char *name) const
{
    PluginMap::const_iterator it = m_plugins.find(name);
    if (it == m_plugins.end())
        return NULL;

    return it->second;
}

vvPlugin *vvPluginList::addPlugin(const char *name, PluginDomain domain)
{
    std::string arg(name);
    std::string n = vvMSController::instance()->syncString(arg);
    if (n != arg)
    {
        std::cerr << "vvPluginList::addPlugin(" << arg << "), but master is trying to load " << n << std::endl;
        abort();
    }
    vvPlugin *m = getPlugin(name);
    if (m == NULL)
    {
        m = loadPlugin(name, domain == Default);
        if (m)
        {
            if (m->init())
            {
                manage(m, domain);
                m->m_initDone = true;
                m->init2();
                m->setTimestep(m_currentTimestep);
                if (domain == Default)
                    updateState();
            }
            else
            {
                if (domain == Default)
                    cerr << "plugin " << name << " failed to initialise" << endl;
                delete m;
                m = NULL;
            }
        }
    }
    auto shared = m_sharedPlugins.find(name);
    if (shared != m_sharedPlugins.end())
    {
        *shared->second = true;
    }
    return m;
}

void vvPluginList::forwardMessage(const covise::DataHandle& dh) const
{
    int headerSize = 2 * sizeof(int);
    if (dh.length() < headerSize)
    {
        cerr << "wrong Message received in vvPluginList::forwardMessage" << endl;
        return;
    }
    const char* buf = dh.data();
    int toWhom = *((int *)buf);
    int type = *(((int *)buf) + 1);
#ifdef BYTESWAP
    byteSwap(toWhom);
    byteSwap(type);
#endif
    if ((toWhom == vvPluginSupport::TO_ALL)
        || (toWhom == vvPluginSupport::TO_ALL_OTHERS)
        || (toWhom == vvPluginSupport::VRML_EVENT))
    {
        message(toWhom, type, dh.length() - headerSize, buf + headerSize);
    }
    else
    {
        const char *name = (((const char *)buf) + headerSize);
        vvPlugin *mod = getPlugin(name);
        if (mod)
        {
            int ssize = (int)( strlen(name) + 1 + (8 - ((strlen(name) + 1) % 8)));
            mod->message(toWhom, type, dh.length() - headerSize - ssize, ((const char *)buf) + headerSize + ssize);
        }
    }
}

void vvPluginList::requestQuit(bool killSession) const
{
    DOALL(plugin->requestQuit(killSession));
}

bool vvPluginList::sendVisMessage(const Message *msg) const
{
    DOALL(if (plugin->sendVisMessage(msg))
          { return true;
          })
    return false;
}

bool vvPluginList::becomeCollaborativeMaster() const
{
    DOALL(if (plugin->becomeCollaborativeMaster())
          { return true;
          })

    return false;
}

covise::Message *vvPluginList::waitForVisMessage(int messageType) const
{
    DOALL(covise::Message *m = plugin->waitForVisMessage(messageType);
          if (m)
          { return m;
    });

    return NULL;
}
/*
void vvPluginList::expandBoundingSphere(vsg::BoundingSphere &bs) const
{
    DOALL(plugin->expandBoundingSphere(bs));
}*/

void vvPluginList::updateState()
{
    //vvTui::instance()->updateState();
    //vvPluginMenu::instance()->updateState();
}
