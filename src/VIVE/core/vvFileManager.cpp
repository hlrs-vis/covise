/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvFileManager.h"
#include "vvHud.h"
#include <vsg/all.h>

#include <cassert>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <locale>
#include <thread>

#include "vvVIVE.h"
#include "vvRegisterSceneGraph.h"
#include "vvSceneGraph.h"
#include "vvViewer.h"
#include "vvTabletUI.h"
#include "vvCommunication.h"
#include "vvConfig.h"
#include "vvIOReader.h"
#include "vvMSController.h"
#include "vvPartner.h"
#include "vvPluginList.h"
#include "vvPluginSupport.h"
#include "PluginMessageTypes.h"
//#include "vvRenderer.h"
#include "vvSidecarConfigBridge.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/FileBrowser.h"
#include "ui/Group.h"
#include "ui/Menu.h"
#include "ui/Owner.h"
#include <config/CoviseConfig.h>
#include <net/covise_host.h>
#include <net/message.h>
#include <util/coFileUtil.h>
#include <util/coHashIter.h>
#include <util/string_util.h>
#include <util/unixcompat.h>
#include <vrb/client/VRBClient.h>
#include <vsg/core/observer_ptr.h>
#include <vsgXchange/all.h>

#include <boost/filesystem/exception.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/locale.hpp>
#include <fcntl.h>

#ifdef HAVE_LIBCURL
#include <HTTPClient/CURL/request.h>
#include <HTTPClient/CURL/methods.h>
#endif

#ifdef __DARWIN_OSX__
#include <Carbon/Carbon.h>
#endif

// undef this to get no START/END messaged
#undef VERBOSE

#ifdef VERBOSE
#define START(msg) fprintf(stderr, "START: %s\n", (msg))
#define END(msg) fprintf(stderr, "END:   %s\n", (msg))
#else
#define START(msg)
#define END(msg)
#endif

using namespace covise;
namespace fs = boost::filesystem;
namespace vive
{

vvFileManager *vvFileManager::s_instance = NULL;


Url::Url(const std::string &url)
{
    if (url.empty())
        return;

    auto it = url.begin();
    // must contain scheme and must begin with an alphabet character
    if (!isalpha(*it))
        return;

    std::locale C("C");
    for ( ; it != url.end(); ++it)
    {
        if (std::isalnum(*it, C))
            continue;
        if (*it == '+')
            continue;
        if (*it == '-')
            continue;
        if (*it == '.')
            continue;
        if (*it == ':')
            break;

        // weird character in URL scheme
        return;
    }
    if (it == url.end())
        return;
#ifdef WIN32
    // probably just a drive letter, not a URL scheme
    if (it - url.begin() <= 1)
        return;
#endif

    m_scheme = std::string(url.begin(), it);
    ++it;

    int numSlash = 0;
    auto authorityBegin = it;
    for ( ; it != url.end(); ++it)
    {
        if (numSlash >= 2)
            break;
        if (*it != '/')
            break;
        ++numSlash;
    }
    if (numSlash >= 2)
    {
        m_haveAuthority = true;
        auto slash = std::find(it, url.end(), '/');
        auto question = std::find(it, url.end(), '?');
        auto hash = std::find(it, url.end(), '#');
        auto end = slash;
        if (hash < end)
            end = hash;
        if (question < end)
            end = question;
        m_authority = decode(std::string(it, end));
        it = end;
    }
    else
    {
        // no authority
        it = authorityBegin;
    }

    auto question = std::find(it, url.end(), '?');
    auto hash = std::find(it, url.end(), '#');
    if (question == url.end())
    {
        m_path = decode(std::string(it, hash), true);
    }
    else
    {
        m_path = decode(std::string(it, question), true);
        it = question;
        ++it;
        hash = std::find(it, url.end(), '#');
        m_query = decode(std::string(it, hash));
    }
    it = hash;
    if (it != url.end())
        ++it;
    m_fragment = decode(std::string(it, url.end()));

    m_valid = true;
}

Url Url::fromFileOrUrl(const std::string &furl)
{
    Url url(furl);
    if (url.valid())
        return url;

    Url file;
    file.m_scheme = "file";
    file.m_path = furl;
    file.m_valid = true;
    return file;
}

std::string Url::decode(const std::string &str, bool path)
{
    return url_decode(str, path);
}

std::string Url::str() const
{
    if (!valid())
        return "(invalid)";

    if (scheme() == "file")
        return path();

    std::string str = scheme();
    str += ":";
    if (m_haveAuthority)
    {
        str += "//";
        str += authority();
    }
    str += path();
    if (!m_query.empty())
    {
        str += "?";
        str += m_query;
    }
    if (!m_fragment.empty())
    {
        str += "#";
        str += m_fragment;
    }

    return str;
}

std::string Url::extension() const
{
    if (!m_valid)
        return std::string();
    if (m_path.empty())
        return std::string();
    auto pos = m_path.find_last_of('.');
    if (pos == std::string::npos)
        return std::string();
    return m_path.substr(pos);
}

bool Url::valid() const
{
    return m_valid;
}

bool Url::isLocal() const
{
    return scheme() == "file";
}

const std::string &Url::scheme() const
{
    return m_scheme;
}

const std::string &Url::authority() const
{
    return m_authority;
}

const std::string &Url::path() const
{
    return m_path;
}

const std::string &Url::query() const
{
    return m_query;
}

const std::string &Url::fragment() const
{
    return m_fragment;
}

Url::Url()
{
}

std::ostream &operator<<(std::ostream &os, const Url &url)
{
    os << url.str();
    return os;
}

std::string shortenUrl(const Url &url, size_t length=20)
{
    return url.str();
}

struct LoadedFile
{
    LoadedFile(const Url &url, ui::Button *button=nullptr, bool isRoot=false)
        : url(url)
        , button(button)
        , isRoot(isRoot)
    {
        if  (button)
        {
            auto n = vvFileManager::instance()->getFileName(url.str());
            if (n.length() == 0)
            {
                n = url.str();
            }
            button->setState(false);
            button->setCallback([this](bool state){
              setVisible(state);
              updateButton();
            });
            updateButton();
        }
    }
    
    ~LoadedFile()
    {
        if (vv->debugLevel(3)) {
            if (node)
                std::cerr << "vvFileManager: LoadedFile: destroy " << url.str() << ", refcount=" << node->referenceCount() << std::endl;
            else
                std::cerr << "vvFileManager: LoadedFile: destroy " << url.str() << ", NO NODE" << std::endl;
        }
    }
    /*
    void objectDeleted(void *d) override
    {
        auto *deleted = static_cast<osg::Referenced *>(d);
        std::cerr << "vvFileManager: LoadedFile: parent of " << url.str() << " deleted" << std::endl;
        parents.erase(static_cast<vsg::Group *>(deleted));
    }*/

    Url url;
    std::shared_ptr<ui::Button> button;

    bool isRoot = false;
    std::string key;
    vsg::ref_ptr<vsg::Node> node;
    std::set<vsg::Group *> parents;
    const FileHandler *handler = nullptr;
    vvIOReader *reader = nullptr;
    vvTUIFileBrowserButton *filebrowser = nullptr;
    int loadCount = 0;

    int numInstances() const
    {
        return loadCount;
    }

    void updateButton() {
        if (!button)
            return;

        auto n = vvFileManager::instance()->getFileName(url.str());
        if (n.length() == 0)
        {
            n = url.str();
        }
        //std::cerr << "updateButton: loadCount=" << loadCount << ", #parents=" << parents.size() << std::endl;
        if (numInstances() > 1)
            n += " (" + std::to_string(numInstances()) + "x)";
        button->setText(n);
    }

    void setVisible(bool state)
    {
        if (state) {
            if (node)
            {
                for (const auto &p : parents) {
                    p->addChild(node);
                    //p->removeObserver(this);
                }
            }
            parents.clear();
        } /*else {
          TODO, store parent  while (node && node->getNumParents() > 0) {
                unsigned i = node->getNumParents() - 1;
                auto p = node->getParent(i);
                parents.emplace(p);
                p->addObserver(this);
                p->removeChild(node);
            }
        }*/
    }

    vsg::ref_ptr<vsg::Node>load();

    vsg::Node *reload()
    {
        std::vector<vsg::ref_ptr<vsg::Group>> parents;
       /* while (node && node->getNumParents() > 0) {
            unsigned i = node->getNumParents() - 1;
            auto p = node->getParent(i);
            parents.emplace_back(p);
            p->removeChild(node);
        }*/

        int lc = loadCount;
        while (loadCount > 0)
            unload();
        assert(node == nullptr);
        node = nullptr;
        while (loadCount < lc)
            node = load();

        for (const auto &p : parents) {
            p->addChild(node);
        }

        return node;
    }

  bool unload()
  {
      if (loadCount <= 0)
          return false;
      --loadCount;
      if (loadCount > 0)
          return true;

      const char *ck = nullptr;
      if (!key.empty())
          ck = key.c_str();

      bool ok = false;
      if (handler)
      {
          if (handler->unloadFile)
          {
              ok = handler->unloadFile(url.str().c_str(), ck) == 0;
          }
         // if (node)
           //   node.release();
      }
      else if (reader)
      {
          ok = reader->unload(node);
      }
      else if (node)
      {
          //parent->removeChild(node);
          //node.release();

          ok = true;
      }

      /*while (node)
      {
          unsigned n = node->getNumParents();
          if (n == 0)
              break;
          node->getParent(n-1)->removeChild(node);
      }*/
      if (vv->debugLevel(3)) {
          if (node)
              std::cerr << "unload: node removed from all parents, refcount="
                        << node->referenceCount() << std::endl;
          else
              std::cerr << "unload: node is NULL" << std::endl;
      }
      node = nullptr;

      return ok;
  }
};

vsg::ref_ptr<vsg::Node> LoadedFile::load()
{
    if (node) {
        ++loadCount;
        return node;
    }

    const char *covise_key = nullptr;
    if (!key.empty())
        covise_key = key.c_str();

    auto adjustedFileName = url.str();
    auto &fb = filebrowser;

    vsg::ref_ptr<vsg::Group> fakeParent = vsg::Group::create();
    if (handler)
    {
        if (vv->debugLevel(3))
            fprintf(stderr, "vvFileManager::loadFile(name=%s)   handler\n", url.str().c_str());
        if (handler->loadUrl)
        {
			if (handler->loadUrl(url, fakeParent, covise_key) < 0)
			{
				if (button)
					button->setState(false);
			}
        }
        else
        {
            if (fb)
            {
                std::string tmpFileName = fb->getFilename(adjustedFileName);
                if (tmpFileName == "")
                    tmpFileName = adjustedFileName;
                std::cerr << "fb: tmpFileName=" << tmpFileName << std::endl;

                if (handler->loadFile)
                    handler->loadFile(tmpFileName.c_str(), fakeParent, covise_key);
            }
            else
            {
                if (handler->loadFile)
                    handler->loadFile(adjustedFileName.c_str(), fakeParent, covise_key);
            }
        }
    }
    else if (reader)
    {
        //fprintf(stderr, "vvFileManager::loadFile(name=%s)  reader\n", fileName);
        std::string filenameToLoad = adjustedFileName;

        if (fb)
        {
            filenameToLoad = fb->getFilename(adjustedFileName);
            if (filenameToLoad == "")
                filenameToLoad = adjustedFileName;
        }

        if (reader->canLoadParts())
        {
            if (vv->debugLevel(3))
                std::cerr << "vvFileManager::loadFile info: loading parts" << std::endl;
            vvFileManager::IOReadOperation op;
            op.filename = filenameToLoad;
            op.reader = reader;
            op.group = fakeParent;
            vvFileManager::instance()->readOperations[reader->getIOHandlerName()].push_back(op);
        }
        else
        {
            if (vv->debugLevel(3))
                std::cerr << "vvFileManager::loadFile info: loading full" << std::endl;
            reader->load(filenameToLoad, fakeParent);
        }
    }
    else
    {
        //fprintf(stderr, "vvFileManager::loadFile(name=%s)   else\n", fileName);
        //(fileName,fileTypeString);
        //obj-Objects must not be rotated
       // don't run optimizer on STL and OBJ files

        std::string tmpFileName = adjustedFileName;
        if (fb)
        {
            tmpFileName = fb->getFilename(adjustedFileName);
        }
        node = vsg::read_cast<vsg::Node>(tmpFileName, vvPluginSupport::instance()->options);
        if (node)
        {
            // vvVIVE::instance()->databasePager->registerPagedLODs(node);
            //node->setNodeMask(node->getNodeMask() & (~Isect::Intersection));
            //if (vv->debugLevel(3))
              //  fprintf(stderr, "vvFileManager::loadFile setting nodeMask of %s to %x\n", node->getName().c_str(), node->getNodeMask());
        }
        else
        {
            if (covise::coFile::exists(adjustedFileName.c_str()))
                std::cerr << "WARNING: Could not load file " << adjustedFileName << std::endl;
        }

        //vvViewer::instance()->Compile();
    }

    if (!node)
    {
        auto n = fakeParent->children.size();
        if (n == 1)
        {
            node = fakeParent->children[0];
        }
        else if (n > 1)
        {
            vsg::Group *g = new vsg::Group();
            node = g;
            for (unsigned int i = 0; i < n; ++i)
            {
                g->addChild(fakeParent->children[i]);
            }
        } 
    }
    std::string n;
    if (node && !node->getValue("name",n))
    {
        node->setValue("name",url.str());
    }

    if (isRoot)
        vvFileManager::instance()->m_loadingFile = nullptr;

    assert(loadCount == 0);
    if (node)
    {
        ++loadCount;
    }

    return node;
}

vsg::ref_ptr<vsg::Node> getNodeIfExists(const std::string &name, const std::string &path)
{
    try
    {
        if (fs::exists(path))
        {
            //auto node = osgDB::readNodeFile(path);
            auto node = vsg::read_cast<vsg::Node>(path, vvPluginSupport::instance()->options);
            if (node)
                node->setValue("name",name);
            else
            {
                std::cerr << "Error loading icon " << name << std::endl;
            }
            return node;
        }

    }
    catch (const std::exception&)
    {
        return nullptr;
    }
    return nullptr;
}

// load an icon file looks in covise/share/covise/icons/$LookAndFeel or covise/share/covise/icons
// returns NULL, if nothing found
vsg::ref_ptr<vsg::Node> vvFileManager::loadIcon(const std::string& filename)
{
    static std::array<const char *, 4> suffixes = {"", ".osg", ".iv", ".obj"};

    static std::array<const char *, 2> rawPrefixes = {
        "share/covise/icons/osg/",
        "share/covise/icons/"};

    static std::array<std::string, rawPrefixes.size() + 1> prefixes = {""};
    static bool init = false;
    if(!init)
    {
        std::string look = coCoviseConfig::getEntry("VIVE.LookAndFeel");
        for (size_t i = 0; i < rawPrefixes.size(); i++)
        {
            auto s = rawPrefixes[i] + (look.empty() ? look : look + "/");
            prefixes[i + 1] = getName(s.c_str());
        }
        init = true;
    }
    for(const auto &prefix : prefixes){
        for(const auto suffix : suffixes)
        {
            auto node = getNodeIfExists(filename, prefix + filename + suffix);
            if(node)
                return node;
        }
    }
    return nullptr;
}

// parmanently loads a texture, looks in covise/icons/$LookAndFeel or covise/icons for filename.rgb
// returns NULL, if nothing found, reuses textures that have already been loaded
vsg::ref_ptr<vsg::Image> &vvFileManager::loadTexture(const char *texture)
{
    START("vvFileManager::loadTexture");

    TextureMap::iterator it = textureList.find(texture);
    if (it != textureList.end())
    {
        if (it->second.get())
        {
            if (vv->debugLevel(4))
                std::cerr << "Reusing texture " << texture << std::endl;
            return it->second;
        }
    }
    const char *name = buildFileName(texture);
    if (name == NULL)
    {
        if (vv->debugLevel(2))
            std::cerr << "New texture " << texture << " - not found" << std::endl;
        //return ;
    }
    else
    {
        if (vv->debugLevel(3))
            std::cerr << "New texture " << texture << std::endl;
    }

    auto image = vsg::read_cast<vsg::Image>(name, vvPluginSupport::instance()->options);
    textureList[texture] = image;

    return textureList[texture];
}

const char *vvFileManager::buildFileName(const char *texture)
{
    const char *name = getName(texture);
    std::string look;
    if (name == NULL)
    {
        look = coCoviseConfig::getEntry("VIVE.LookAndFeel");
        if (!look.empty())
        {
            char *fn = new char[strlen(texture) + strlen(look.c_str()) + 50];
            sprintf(fn, "share/covise/icons/%s/%s.rgb", look.c_str(), texture);
            name = getName(fn);
            delete[] fn;
        }
    }
    if (name == NULL)
    {
        char *fn = new char[strlen(texture) + 50];
        sprintf(fn, "share/covise/icons/%s.rgb", texture);
        name = getName(fn);
        delete[] fn;
    }

    return name;
}

bool vvFileManager::fileExist(const char *fileName)
{
	try
	{
		return fs::exists(fileName);
	}
	catch (const std::exception&)
	{
		return false;
	}
	
}

bool vvFileManager::fileExist(const std::string& fileName)
{
	return fileExist(fileName.c_str());
}

vsg::ref_ptr<vsg::Node> vvFileManager::loadFile(const char *fileName, vvTUIFileBrowserButton *fb, vsg::Group *parent, const char *covise_key)
{
    START("vvFileManager::loadFile");
    if (!fileName || strcmp(fileName, "") == 0)
    {
        return nullptr;
    }
    std::string validFileName(fileName);
    if (!parent)
        parent = vv->getObjectsRoot();

	convertBackslash(validFileName);

    auto duplicate = m_files.find(validFileName);
    if (duplicate != m_files.end())
    {
        auto node = duplicate->second->load();
        duplicate->second->updateButton();
        parent->children.push_back(node);
        return node;
    }

    std::string key;
    if (fb)
    {
        //Store filename associated with corresponding fb instance
        key = validFileName;
        fileFBMap[key] = fb;
    }
	//url to local file
    Url url = Url::fromFileOrUrl(validFileName);
    if (!url.valid())
    {
        std::cerr << "failed to parse URL " << validFileName << std::endl;
        return  nullptr;
    }
    if (vv->debugLevel(3))
        std::cerr << "Loading " << url.str() << std::endl;

    std::string adjustedFileName;
    if (url.scheme() == "vive")
    {
        if (url.authority() == "plugin")
        {
            vvPluginList::instance()->addPlugin(url.path().c_str()+1);
        }
        else if (url.authority() == "terrain")
        {
            vvPlugin* plugin = vvPluginList::instance()->addPlugin("GeoData");
            if (plugin)
            {
                std::string terrainFile = url.str();
                plugin->message(vvPluginSupport::TO_ALL, PluginMessageTypes::LoadTerrain, (int)terrainFile.size() + 1, terrainFile.c_str());
            }
            else
            {
                adjustedFileName = url.str();
            }
        }
        else if (url.authority() == "sky")
        {
            vvPlugin* plugin = vvPluginList::instance()->addPlugin("GeoData");
            if (plugin)
            {
                std::string skyNumber = url.str();
                plugin->message(vvPluginSupport::TO_ALL, PluginMessageTypes::setSky, (int)skyNumber.size() + 1, skyNumber.c_str());
            }
        }
        return nullptr;
    }
    else if (url.scheme() == "file")
    {
        adjustedFileName = url.path();
        if (vv->debugLevel(3))
            std::cerr << " New filename: " << adjustedFileName << std::endl;
    }
    else
    {
        adjustedFileName = url.str();
    }


    bool isRoot = m_loadingFile==nullptr;
    ui::Button *button = nullptr;
    if (vv && isRoot)
    {
		std::string relPath(adjustedFileName);
		makeRelativeToSharedDataLink(relPath);
		button = new ui::Button(m_fileGroup, "File" + reduceToAlphanumeric(relPath)+std::to_string(uniqueNumber));
        uniqueNumber++;
    }
	if (isRoot)
	{
		//if file is not shared, add it to the shared filePaths list
		fileOwnerList v = m_sharedFiles;
		std::string pathIdentifier(adjustedFileName);
		makeRelativeToSharedDataLink(pathIdentifier);
		bool found = false;
		for (auto p : v)
		{
			if (p.first == pathIdentifier)
			{
				found = true;
				break;
			}
		}


		if (!found)
		{
			v.push_back(fileOwner(pathIdentifier, vvCommunication::instance()->getID()));
			m_sharedFiles = v;
		}
	}
    auto fe = new LoadedFile(url, button, isRoot);
    if (covise_key)
        fe->key = covise_key;
    fe->filebrowser = fb;
    if (!vvVIVE::instance()->visPlugin() && !m_settings && fe->url.valid() && fe->url.isLocal())
    {
        std::cerr << "Sidecar file for " << fe->url.str() << std::endl;
        m_settings = std::make_unique<vvSidecarConfigBridge>(fe->url.str(), vvMSController::instance()->isMaster());
        vv->m_config.setWorkspaceBridge(m_settings.get());
        m_mainFile = fe->url.str();
    }

    vvVIVE::instance()->hud->setText2("loading");

#ifdef WIN32
    std::string utf8_filename = boost::locale::conv::to_utf<char>(validFileName, "ISO-8859-1");
#else
    std::string utf8_filename = validFileName;
#endif
    vvVIVE::instance()->hud->setText3(utf8_filename);
    vvVIVE::instance()->hud->redraw();

    if (isRoot)
    {
        m_loadingFile = fe;

        if(viewPointFile == "" && url.isLocal())
        {
            const char *ext = strchr(url.path().c_str(), '.');
            if(ext)
            {
                viewPointFile = validFileName;
                std::string::size_type pos = viewPointFile.find_last_of('.');
                viewPointFile = viewPointFile.substr(0,pos);
                viewPointFile+=".vwp";
            }
        }
    }

    /// read the 1st line of file and try to guess the type
    std::string fileTypeString = findFileExt(url);
    const FileHandler *handler = findFileHandler(url.path().c_str());
    vvIOReader *reader = findIOHandler(adjustedFileName.c_str());
    if (!handler && !fileTypeString.empty())
        handler = findFileHandler(fileTypeString.c_str());
	//vrml will remote fetch missing files itself
	std::string xt = url.extension();
	
	std::vector<std::string> vrmlExtentions{ "x3dv", "wrl", "wrz" };
    const char *ive = ".ive";
	std::string lowXt(xt);
	std::transform(xt.begin(), xt.end(), lowXt.begin(), ::tolower);
    auto urlStr = url.str();
	std::transform(urlStr.begin(), urlStr.end(), urlStr.begin(), ::tolower);

	bool isVRML = false;
	for (auto ext : vrmlExtentions)
	{
        size_t extlen = ext.size() + strlen(ive);
		if (("." + ext) == lowXt ||(urlStr.size()> extlen) && (urlStr.substr(urlStr.size() - extlen) == ext + ive))
		{
			isVRML = true;
			break;
		}
    }
    
    if (!isVRML)
    {
        std::string filename = findOrGetFile(adjustedFileName);
		if (filename.length() == 0)
		{
            delete fe;
			return nullptr;
		}
		fe->url = Url::fromFileOrUrl(filename);

	}
    fe->handler = handler;
    fe->reader = reader;
    
	if (fe->button)
	{
		fe->button->setShared(true);
		fe->button->setState(true);
	}

    auto node = fe->load();
    fe->updateButton();
    if (node != NULL)
    {
        fe->filebrowser = nullptr;

        parent->addChild(node);

        if (isRoot) {
            m_files[validFileName] = fe;

            m_lastFile = fe;

           /* vvRegisterSceneGraph::instance()->registerNode(node,
                                                           parent->getName());*/
        }
    }

    if (isRoot)
    {
        //vvViewer::instance()->forceCompile();
  
		if (node)
            vvVIVE::instance()->hud->setText2("done loading");
        else
            vvVIVE::instance()->hud->setText2("failed to load");
        vvVIVE::instance()->hud->redraw();
    }
    else
    {
        delete fe;
    }

    this->fileFBMap.erase(key);
    return node;
}

vsg::ref_ptr<vsg::Node> vvFileManager::replaceFile(const char *fileName, vvTUIFileBrowserButton *fb, vsg::Group *parent, const char *covise_key)
{
    return loadFile(fileName, fb, parent, covise_key);
}

struct Magic
{
    const char *text;
    const char *format;
};

static const int numMagic = 4;
static Magic magic[numMagic] = {
    { "VRML V1.", "iv" },
    { "INVENTOR", "iv" },
    { "#VRML V2", "wrl" },
    { "#X3D V3", "wrl" }
};

std::string vvFileManager::findFileExt(const Url &url)
{
    if (url.scheme() == "file")
    {
        const char *filename = url.path().c_str();
        FILE *infile = fopen(filename, "r");
        if (infile)
        {

            char inbuffer[128], upcase[128];
            if (fgets(inbuffer, 128, infile) == NULL)
            {
                if (vv->debugLevel(3))
                    std::cerr << "vvFileManager::findFileExt: fgets failed" << std::endl;
            }

            fclose(infile);

#ifdef VERBOSE
            cerr << " #### filename='" << filename
                 << "'   header='" << inbuffer
                 << "'" << endl;
#endif

            inbuffer[127] = '\0';
            char *iPtr = inbuffer;
            char *oPtr = upcase;

            while (*iPtr)
            {
                if (*iPtr >= 'a' && *iPtr <= 'z')
                    *oPtr = *iPtr - 'a' + 'A';
                else
                    *oPtr = *iPtr;
                iPtr++;
                oPtr++;
            }
            *oPtr = '\0';
            //cerr << upcase << endl;

            for (int i = 0; i < numMagic; i++)
            {
                if (strstr(upcase, magic[i].text))
                {
#ifdef VERBOSE
                    cerr << "Identified file " << filename
                         << " as " << magic[i].format << endl;
#endif
                    return magic[i].format;
                }
            }
        }
    }
    const auto &path = url.path();
    /* look for final "." in filename */
    const char *ext = strchr(path.c_str(), '.');
    /* no dot, assume it's just the extension */
    if (ext == NULL)
        return std::string();
    else
        /* advance "ext" past the period character */
        ++ext;
    // get rid of uppercase/lowercase endings
    while (ext)
    {
        // VRML
        if (0 == strcasecmp(ext, "vrml"))
            return "wrl";
        if (0 == strcasecmp(ext, "wrl"))
            return "wrl";
        if (0 == strcasecmp(ext, "wrl.gz"))
            return "wrl";
        if (0 == strcasecmp(ext, "wrz"))
            return "wrl";

        // X3DV
        if (0 == strcasecmp(ext, "x3dv"))
            return "wrl";
        if (0 == strcasecmp(ext, "x3dv.gz"))
            return "wrl";
        if (0 == strcasecmp(ext, ".x3dvz"))
            return "wrl";

        // volume data
        if (0 == strcasecmp(ext, "avf"))
            return "avf";
        if (0 == strcasecmp(ext, "xvf"))
            return "xvf";
        if (0 == strcasecmp(ext, "rvf"))
            return "rvf";

        // others
        if (0 == strcasecmp(ext, "iv"))
            return "iv";
        if (0 == strcasecmp(ext, "ive"))
            return "ive";
        if (0 == strcasecmp(ext, "osg"))
            return "osg";
        if (0 == strcasecmp(ext, "obj"))
            return "obj";
        if (0 == strcasecmp(ext, "jt"))
            return "jt";

        const char *newext = strchr(ext, '.');
        if (newext == NULL)
            return ext;
        else
            ext = newext + 1;
    }
    return ext;
}

void vvFileManager::reloadFile()
{
    START("vvFileManager::reloadFile");
    if (m_lastFile)
    {
        std::cerr << "reloading " << m_lastFile->url << std::endl;
        m_loadingFile = m_lastFile;
        m_lastFile->reload();
        m_loadingFile = nullptr;
    }
    else
    {
        std::cerr << "nothing to reload" << std::endl;
    }
}

void vvFileManager::unloadFile(const char *file)
{
    START("vvFileManager::unloadFile");
    if (file)
    {
        //std::cerr << "vvFileManager::unloadFile: file=" << file << std::endl;
        std::string validFileName(file);
        convertBackslash(validFileName);
        auto it = m_files.find(validFileName);
        if (it == m_files.end())
        {
            std::cerr << validFileName << " not loaded" << std::endl;
            return;
        }

        auto &fe = it->second;
        //VRRegisterSceneGraph::instance()->unregisterNode(fe->node, parent->getName());
        if (!fe->parents.empty())
        {
            fe->setVisible(true); // otherwise invalid parent nodes would stick around
            if (fe->button)
                fe->button->setState(true);
        }

        if (!fe->unload())
            std::cerr << "unloading " << fe->url << " failed" << std::endl;
        fe->updateButton();

        //std::cerr << "unloaded " << fe->url << ": #instances=" << fe->numInstances() << std::endl;

        if (fe->numInstances() == 0)
        {
            if (m_lastFile == fe)
                m_lastFile = nullptr;
            delete fe;
            m_files.erase(it);
        }
    }
    else
    {
        m_lastFile = nullptr;
        for (auto &fe: m_files)
        {
            while (fe.second->unload())
                ;
            fe.second->updateButton();
            delete fe.second;
        }
        m_files.clear();
    }

    if (m_files.empty())
    {
        if (m_settings)
        {
            vv->m_config.removeWorkspaceBridge(m_settings.get());
            m_settings.reset();
            m_mainFile.clear();
        }
    }
}

vvFileManager *vvFileManager::instance()
{
    if (!s_instance)
        s_instance = new vvFileManager;
    return s_instance;
}

vvFileManager::vvFileManager()
    : fileHandlerList()
    , m_sharedFiles("vvFileManager_filePaths", fileOwnerList(), vrb::ALWAYS_SHARE)
    , remoteFetchPathTmp((fs::temp_directory_path() / ("remoteFetch_" + covise::Host::getUsername())).string())
{
    START("vvFileManager::vvFileManager");
    /// path for the viewpoint file: initialized by 1st param() call
    assert(!s_instance);
    getSharedDataPath();
	//register my files with my ID
	vvCommunication::instance()->subscribeNotification(vvCommunication::Notification::Connected, [this]() {
		fileOwnerList files = m_sharedFiles.value();
		auto path = files.begin();
		while (path != files.end())
		{
			path->second = vvCommunication::instance()->getID();
			++path;
		}
		m_sharedFiles = files;
    });

    /*osgDB::Registry::instance()->addFileExtensionAlias("gml", "citygml");
    osgDB::Registry::instance()->addFileExtensionAlias("3mxb", "3mx");*/

    /*options = new osgDB::ReaderWriter::Options;
    options->setOptionString(coCoviseConfig::getEntry("options", "VIVE.File"));
    osgDB::Registry::instance()->setOptions(options);*/
    remoteFetchHashPrefix = coCoviseConfig::isOn("hash", "System.VRB.RemoteFetch", true, nullptr);
    if (vvVIVE::instance() && vvVIVE::instance()->useVistle())
    {
	remoteFetchEnabled = false;
    }
    else
    {
	remoteFetchEnabled = coCoviseConfig::isOn("value", "System.VRB.RemoteFetch", false, nullptr);
    }
    std::string path = coCoviseConfig::getEntry("path", "System.VRB.RemoteFetch");
    path = resolveEnvs(path);
    fs::path p{path};
    if (path.empty())
        p = fs::temp_directory_path() / covise::Host::getUsername() / "cover";

    remoteFetchPath = p.string();
    std::cerr << "remotefech path = " << remoteFetchPath << std::endl;
    if(remoteFetchEnabled)
        checkRemoteFetchDirShared();
    //clear tmp dir to ensure same state on all slaves
    if(fs::exists(remoteFetchPathTmp)) 
        fs::remove_all(remoteFetchPathTmp);
}
void vvFileManager::initUI()
{

    if (vv) {
        m_owner.reset(new ui::Owner("FileManager", vv->ui));

        auto fileOpen = new ui::FileBrowser("OpenFile", m_owner.get());
        fileOpen->setText("Open");
        fileOpen->setFilter(getFilterList());
        if (!vv->fileMenu)
            vv->initUI();
        if (vv->fileMenu)
        {
            vv->fileMenu->add(fileOpen);
            fileOpen->setCallback([this](const std::string& file) {
                loadFile(file.c_str());
                });
            m_sharedFiles.setUpdateFunction([this](void) {loadPartnerFiles(); });
            m_fileGroup = new ui::Group("LoadedFiles", m_owner.get());
            m_fileGroup->setText("Files");
            vv->fileMenu->add(m_fileGroup);

            auto fileReload = new ui::Action("ReloadFile", m_owner.get());
            vv->fileMenu->add(fileReload);
            fileReload->setText("Reload file");
            fileReload->setCallback([this]() {
                reloadFile();
                });

            auto fileSave = new ui::FileBrowser("SaveFile", m_owner.get(), true);
            fileSave->setText("Save");
            fileSave->setFilter(getWriteFilterList());
            vv->fileMenu->add(fileSave);
            fileSave->setCallback([this](const std::string& file) {
                if (vvMSController::instance()->isMaster())
                    vvSceneGraph::instance()->saveScenegraph(file);
                });

            vv->getUpdateManager()->add(this);
            vvCommunication::instance()->initVrbFileMenu();
        }
    }
}

void vvFileManager::createRemoteFetchDir()
{
if (!remoteFetchDirExists)
{
    try
    {
        if(!fs::exists(remoteFetchPath))
            fs::create_directories(remoteFetchPath);
        remoteFetchDirExists = true;
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Could not create directory: " << remoteFetchPath << " : " << e.what() << std::endl;
    }
}

}
vvFileManager::~vvFileManager()
{
    START("vvFileManager::~vvFileManager");
    if (vv->debugLevel(2))
        fprintf(stderr, "delete vvFileManager\n");

#if 0
    for (auto &f: m_files)
    {
        delete f.second;
    }
    m_files.clear();
#endif

    vv->getUpdateManager()->remove(this);

    s_instance = NULL;
}

//=====================================================================
//
//=====================================================================
const char *vvFileManager::getName(const char *file)
{
    START("vvFileManager::getName");
    static char *buf = NULL;
    static size_t buflen = 0;

    if (file == NULL)
        return NULL;
    else if (file[0] == '\0')
        return NULL;

    if ((buf == NULL) || (buflen < (int)(strlen(file) + 20)))
    {
        buflen = strlen(file) + 100;
        delete[] buf;
        buf = new char[buflen];
    }
    sprintf(buf, "%s", file);
    FILE *fp = ::fopen(buf, "r");
    if (fp != NULL)
    {
        fclose(fp);
        return buf;
    }

    char *covisepath = getenv("COVISE_PATH");
    if (!covisepath)
    {
        if (vv->debugLevel(3))
            cerr << "ERROR: COVISE_PATH not defined\n";
        return NULL;
    }
    if ((buf == NULL) || (buflen < (int)(strlen(covisepath) + strlen(file) + 20)))
    {
        buflen = strlen(covisepath) + strlen(file) + 100;
        delete[] buf;
        buf = new char[buflen];
    }
    char *coPath = new char[strlen(covisepath) + 1];
    strcpy(coPath, covisepath);
#ifdef _WIN32
    char *dirname = strtok(coPath, ";");
#else
    char *dirname = strtok(coPath, ":");
#endif
    while (dirname != NULL)
    {
        sprintf(buf, "%s/%s", dirname, file);
        boost::filesystem::path p(buf);
        if(boost::filesystem::exists(p))
        {
            delete[] coPath;
            return buf;
        }
#if 0
        for (int i = strlen(dirname) - 2; i > 0; i--)
        {
            if (dirname[i] == '/')
            {
                dirname[i] = '\0';
                break;
            }
            else if (dirname[i] == '\\')
            {
                dirname[i] = '\0';
                break;
            }
        }
        sprintf(buf, "%s/%s", dirname, file);
        fp = ::fopen(buf, "r");
        if (fp != NULL)
        {
            fclose(fp);
            delete[] coPath;
            return buf;
        }
#endif
#ifdef _WIN32
        dirname = strtok(NULL, ";");
#else
        dirname = strtok(NULL, ":");
#endif
    }
    delete[] coPath;
    buf[0] = '\0';
    return NULL;
}

bool vvFileManager::makeRelativeToSharedDataLink(std::string & fileName)
{
    try
    {
        if (fs::exists(fileName))
        {
            return makeRelativePath(fileName, m_sharedDataLink);
        }
    }
    catch(fs::filesystem_error)
    {
        return false;
    }
	return false;
	
}

bool vvFileManager::isInTmpDir(const std::string& fileName)
{
	std::string f(fileName);
	return makeRelativePath(f, fs::temp_directory_path().string());
}

bool vvFileManager::isInSharedDir(const std::string& fileName)
{
	std::string f(fileName);
    return (remoteFetchPathShared && makeRelativePath(f, remoteFetchPath)) || isInTmpDir(fileName);
}

void vvFileManager::checkRemoteFetchDirShared()
{
    auto testFile = remoteFetchPath + "/coviseSharedFileSystemTest";
    vvMSController *ms = vvMSController::instance();
    createRemoteFetchDir();
    if (ms->isMaster())
        std::fstream s(testFile, std::ios_base::out);
    remoteFetchPathShared = true;
    if (ms->isSlave())
    {
        bool shared;
        try
        {
            shared = fs::exists(testFile);
        }
        catch (fs::filesystem_error)
        {
            shared = false;
        }
        ms->sendMaster(&shared, sizeof(bool));
    }
    else if(ms->getNumSlaves() > 0)
    {
        vvMSController::SlaveData sd(sizeof(bool));
        ms->readSlaves(&sd);
        bool allSame = true;
        for (size_t i = 1; i < sd.data.size(); i++)
        {
            if(*(bool*)sd.data[i-1] != *(bool*)sd.data[i])
            {
                allSame = false;
                break;
            }
        }
        if(!allSame)
        {
            remoteFetchEnabled = false;
            std::cerr << "remote fetch dir " << remoteFetchPath << " is shared with some slaves but not all." << std::endl <<
            "This is not supported and remote fetch is turned off " << std::endl;
        }else
        {
            remoteFetchPathShared = *(bool *)sd.data[0];
        }
    }
    remoteFetchEnabled = ms->syncBool(remoteFetchEnabled);
    remoteFetchPathShared = ms->syncBool(remoteFetchPathShared);
    if(ms->isMaster())
        fs::remove(testFile);
}

bool vvFileManager::makeRelativePath(std::string& fileName,  const std::string& basePath)
{
  if(basePath.empty())
      return true;
  auto fn = fs::relative(fs::path(fileName), fs::path(basePath)).string();
  if(fn.size() > 3 || (fn[0] == '.' && fn[1] == '.'))
    return false;
  fileName = fn;
  return true;
}

void vvFileManager::fetchObjMaterials(const std::string & localPath, const std::string &remotePath, int fileOwner)
{
    if (fs::path(localPath).extension().string() == ".obj" )
    {
        auto f = fstream(localPath);
        if(!f.is_open())
        {
            std::cerr << "fetchObjMaterials: failed to open file  " << localPath << std::endl;
            return;
        }
        std::string line;
        while (std::getline(f, line))
        {
            std::istringstream iss(line);
            std::string keyword;
            iss >> keyword;
            if (keyword != "mtllib")
                continue;
            while(!iss.eof())
            {
                std::string mat;
                iss >> mat;
                if(mat[0] == '/' || remotePath.find('/') == std::string::npos)//absolute path
                    remoteFetch(mat, fileOwner);
                else if(remotePath.find('/') != std::string::npos){
                    auto dir = remotePath.substr(0, remotePath.find_last_of('/') + 1);
                    remoteFetch(dir + mat, fileOwner);
                }
            }

        }

    }
}

std::string getRemoteFetchHashPrefix(const std::string& filePath, bool doSth)
{
    return doSth ? std::to_string(std::hash<std::string>{}(fs::path{filePath}.parent_path().string())) + "/" : "";
}

std::string vvFileManager::findFile(const std::string &fileName)
{
    if(fileName.empty())
        return "";
    const std::array<fs::path, 6> searchLocations = {
        fs::path(fs::path{fileName}),
        fs::path(fs::current_path() / fileName),
        fs::path(fs::path{m_sharedDataLink} / fileName),
        fs::path(fs::path{remoteFetchPath} /  getRemoteFetchHashPrefix(fileName, remoteFetchHashPrefix) / getFileName(fileName)),
        fs::path(fs::path{remoteFetchPathTmp} /  getRemoteFetchHashPrefix(fileName, remoteFetchHashPrefix) / getFileName(fileName)),
    };
    for(const auto &p : searchLocations)
    {
        try
        {
            if (fs::exists(p))
            {
                return p.string();
            }
        }
        catch (fs::filesystem_error)
        {
            return "";
        }
    }
    return "";
};

std::string vvFileManager::findOrGetFile(const std::string& filePath,  int where)
{

    auto localPath = findFile(filePath);
    vvMSController *ms = vvMSController::instance();

    assert(ms->syncBool(localPath.empty()) == (localPath.empty()) && "findOrGetFileSyncError");

    if (localPath.empty())
    {
        if(filePath.rfind("http", 0) != std::string::npos|| filePath.rfind("https", 0)!= std::string::npos)
        {
            return httpFetch(filePath);
        }
        else if (remoteFetchEnabled && vv->isVRBconnected())  
        {
            int fileOwner = where == 0 ? guessFileOwner(filePath) : where;
            auto path = remoteFetch(filePath, fileOwner);
            if (fileExist(path))
            {
                fetchObjMaterials(path, filePath, fileOwner);
            }
            return path;
        }
    }
	return localPath;
}

std::string vvFileManager::getFontFile(const char *fontname)
{
    const std::string fallback("NotoSans-Regular.ttf");
    const std::string fontpath("share/covise/fonts/");

    if (fontname)
    {
        const char *name = getName(fontname);
        if (name)
            return name;
        name = getName((fontpath + fontname).c_str());
        if (name)
            return name;
    }

    if (m_defaultFontFile.empty()) {
        std::string fontFile = coCoviseConfig::getEntry("value", "VIVE.Font", fontpath + fallback);
        if (const char *name = getName(fontFile.c_str())) {
            m_defaultFontFile = name;
        } else if (const char *name = getName((fontpath + fallback).c_str())) {
            m_defaultFontFile = name;
        }
    }

    return m_defaultFontFile;
}

vsg::ref_ptr<vsg::Font> vvFileManager::loadFont(const char *fontname)
{
    if (fontname == nullptr)
    {
        if (defaultFont.get()==nullptr)
        {
            defaultFont = vsg::read_cast<vsg::Font>(getFontFile("times.vsgb"), vvPluginSupport::instance()->options);
        }
        return defaultFont;
    }
    return vsg::read_cast<vsg::Font>(getFontFile(fontname), vvPluginSupport::instance()->options);
}

int vvFileManager::coLoadFontDefaultStyle()
{

    return 0;
}

std::string vvFileManager::getFilterList()
{
    std::string extensions;
    for (FileHandlerList::iterator it = fileHandlerList.begin();
         it != fileHandlerList.end();
         ++it)
    {
        extensions += "*.";
        extensions += (*it)->extension;
        extensions += ";";
    }
    extensions += "*.wrl;";
    extensions += "*.osg *.ive;";
    extensions += "*.osgb *.osgt *.osgx;";
    extensions += "*.obj;";
    extensions += "*.stl;";
    extensions += "*.ply;";
    extensions += "*.iv;";
    extensions += "*.dxf;";
    extensions += "*.3ds;";
    extensions += "*.flt;";
    extensions += "*.dae;";
    extensions += "*.md2;";
    extensions += "*.geo;";
    extensions += "*";

    return extensions;
}

std::string vvFileManager::getWriteFilterList()
{
    std::string extensions;
    extensions += "*.osg;";
    extensions += "*.ive;";
    extensions += "*.osgb;";
    extensions += "*.osgt;";
    extensions += "*.osgx;";
    extensions += "*.obj;";
    extensions += "*.stl;";
    extensions += "*.3ds;";
    extensions += "*.iv;";
    extensions += "*.dae;";
    extensions += "*";

    return extensions;
}


const FileHandler *vvFileManager::getFileHandler(const char *extension)
{
    for (FileHandlerList::iterator it = fileHandlerList.begin();
         it != fileHandlerList.end();
         ++it)
    {
        if (!strcasecmp(extension, (*it)->extension))
            return *it;
    }

    return NULL;
}

const FileHandler *vvFileManager::findFileHandler(const char *pathname)
{
    std::vector<const char *> extensions;
    if (const char *p = strrchr(pathname, '/'))
    {
        extensions.push_back(p+1);
    }
    else
    {
        extensions.push_back(pathname);
    }
    for (const char *p = strchr(pathname, '.'); p; p = strchr(p, '.'))
    {
        ++p;
        const char *extension = p;
        extensions.push_back(extension);
    }

    for (auto extension: extensions)
    {
        for (FileHandlerList::iterator it = fileHandlerList.begin();
                it != fileHandlerList.end();
                ++it)
        {
            if (!strcasecmp(extension, (*it)->extension))
                return *it;
        }

        size_t extlen = strlen(extension);
        char *cEntry = new char[40 + extlen];
        char *lowerExt = new char[extlen + 1];
        for (size_t i = 0; i < extlen; i++)
        {
            lowerExt[i] = tolower(extension[i]);
            if (lowerExt[i] == '.')
                lowerExt[i] = '_';
        }
        lowerExt[extlen] = '\0';

        sprintf(cEntry, "VIVE.FileManager.FileType:%s", lowerExt);
        string plugin = coCoviseConfig::getEntry("plugin", cEntry);
        delete[] cEntry;
        delete[] lowerExt;
        if (plugin.size() > 0)
        { // load the appropriate plugin and give it another try
            vvPluginList::instance()->addPlugin(plugin.c_str());
            for (FileHandlerList::iterator it = fileHandlerList.begin();
                    it != fileHandlerList.end();
                    ++it)
            {
                if (!strcasecmp(extension, (*it)->extension))
                {
                    if (vv->debugLevel(2))
                        fprintf(stderr, "vvFileManager::findFileHandler(extension=%s), using plugin %s\n", extension, plugin.c_str());
                    return *it;
                }
            }
        }
    }

    return NULL;
}

vvIOReader *vvFileManager::findIOHandler(const char *pathname)
{
    vvIOReader *best = NULL;
    size_t maxmatch = 0;
    const std::string file(pathname);

    for (IOReaderList::iterator it = ioReaderList.begin();
         it != ioReaderList.end(); ++it)
    {
        typedef std::list<std::string> ExtList;
        const ExtList &extlist = (*it)->getSupportedReadFileExtensions();
        for (ExtList::const_iterator it2 = extlist.begin();
                it2 != extlist.end();
                ++it2)
        {
            const std::string &ext = *it2;
            if (ext.length() > file.length())
                continue;
            if (std::equal(ext.rbegin(), ext.rend(), file.rbegin()))
            {
                if (ext.length() > maxmatch)
                {
                    maxmatch = ext.length();
                    best = *it;
                }
            }

        }
    }
    return best;
}

int vvFileManager::registerFileHandler(const FileHandler *handler)
{
    if (getFileHandler(handler->extension))
        return -1;

    fileHandlerList.push_back(handler);
    return 0;
}

int vvFileManager::registerFileHandler(vvIOReader *handler)
{
    //if(getFileHandler(handler->extension))
    //   return -1;

    if (vv->debugLevel(3))
        std::cerr << "vvFileManager::registerFileHandler info: registering " << handler->getIOHandlerName() << std::endl;

    this->ioReaderList.push_back(handler);
    return 0;
}

int vvFileManager::unregisterFileHandler(const FileHandler *handler)
{
    if (!handler || !handler->extension)
        return -1;

    for (FileHandlerList::iterator it = fileHandlerList.begin();
         it != fileHandlerList.end();
         ++it)
    {
        if (strcmp(handler->extension, (*it)->extension) == 0)
        {
            const FileHandler *p = *it;
            if (p->loadFile == handler->loadFile
                && p->unloadFile == handler->unloadFile)
            {
                for (auto it = m_files.begin(), next = it; it != m_files.end(); it = next)
                {
                    auto &fe = it->second;
                    if (m_lastFile == fe)
                        m_lastFile = nullptr;
                    if (fe->handler == p)
                    {
                        while (fe->unload())
                            ;
                        fe->updateButton();
                        delete fe;
                        next = m_files.erase(it);
                    }
                    next = it;
                    ++next;
                }

                fileHandlerList.erase(it);
                return 0;
            }
        }
    }
    return -1;
}

int vvFileManager::unregisterFileHandler(vvIOReader *handler)
{
    if (handler == 0)
        return -1;

    if (vv->debugLevel(3))
        std::cerr << "vvFileManager::unregisterFileHandler info: unregistering " << handler->getIOHandlerName() << std::endl;

    for (IOReaderList::iterator it = ioReaderList.begin();
         it != ioReaderList.end(); ++it)
    {
        if (*it == handler)
        {
            ioReaderList.erase(it);
            return 0;
        }
    }
    return -1;
}

vvTUIFileBrowserButton *vvFileManager::getMatchingFileBrowserInstance(std::string keyFileName)
{
    vvTUIFileBrowserButton *fbo = NULL;
    std::map<string, vvTUIFileBrowserButton *>::iterator itr = this->fileFBMap.find(keyFileName);

    if (itr != this->fileFBMap.end())
        fbo = itr->second;
    else
    {
        if (this->IsDefFBSet())
            fbo = this->mDefaultFB;
        else
            fbo = NULL;
    }

    return fbo;
}

bool vvFileManager::IsDefFBSet()
{
    if (this->mDefaultFB)
        return true;
    else
        return false;
}

void vvFileManager::SetDefaultFB(vvTUIFileBrowserButton *fb)
{
    this->mDefaultFB = fb;
}

void vvFileManager::setViewPointFile(const std::string &file)
{
    viewPointFile = file;
}

std::string vvFileManager::getViewPointFile()
{
    return viewPointFile;
}

bool vvFileManager::update()
{
    for (ReadOperations::iterator op = this->readOperations.begin(); op != this->readOperations.end(); ++op)
    {
        std::string reader = op->first;
        if (this->readOperations[reader].empty())
        {
            this->readOperations.erase(op);
            continue;
        }

        IOReadOperation readOperation = this->readOperations[reader].front();

        vvIOReader::IOStatus status = readOperation.reader->loadPart(readOperation.filename, readOperation.group);

        if (status == vvIOReader::Finished)
        {
            if (vv->debugLevel(3))
                std::cerr << "vvFileManager::update info: finished loading " << readOperation.filename << std::endl;
            readOperations[reader].pop_front();
        }
        else if (status == vvIOReader::Failed)
        {
            if (vv->debugLevel(3))
                std::cerr << "vvFileManager::update info: failed loading " << readOperation.filename << std::endl;
            readOperations[reader].pop_front();
        }
        else
        {
            if (vv->debugLevel(3))
                std::cerr << "vvFileManager::update info: loading " << readOperation.filename << " (" << readOperation.reader->getIOProgress() << ")" << std::endl;
        }
    }
    return true;
}

void vvFileManager::loadPartnerFiles()
{
    //load and unload existing files
    std::set<std::string> alreadyLoadedFiles;
    for (auto myFile : m_files)
    {
        auto shortPath = myFile.first;
		auto fileName = getFileName(shortPath);
        makeRelativeToSharedDataLink(shortPath);
        alreadyLoadedFiles.insert(fileName);
		bool found = false;
		for (auto p : m_sharedFiles.value())
		{
			auto pFileName = getFileName(p.first);
			if (pFileName == fileName)
			{
				//myFile.second->load();
				found = true;
				break;
			}
		}
		if (!found)
		{
			//unloadFile(myFile.first.c_str());
		}
    }

    //load new partner files
    std::set<std::string> newFiles;
	for (auto theirFile : m_sharedFiles.value())
	{
		if (alreadyLoadedFiles.find(getFileName(theirFile.first)) == alreadyLoadedFiles.end())
		{
			loadFile(theirFile.first.c_str());
		}
	}
}

void vvFileManager::getSharedDataPath()
{
    char *covisepath = getenv("COVISE_PATH");
#ifdef WIN32
    std::vector<std::string> p = split(covisepath, ';');
#else
    std::vector<std::string> p = split(covisepath, ':');
#endif
    for (auto path : p)
    {
		convertBackslash(path);
		size_t delimiter = path.rfind('/');
		std::string link = path.erase(delimiter, path.length()- delimiter) + "/sharedData";
        if (fileExist(link))
        {
			m_sharedDataLink = link;
			return;
        }
    }
}

void vvFileManager::convertBackslash(std::string & path)
{
    for (char &c : path)
      c == '\\' ? c = '/' : c = c;
}

std::string vvFileManager::remoteFetch(const std::string& filePath, int fileOwner)
{

	if(!vv->isVRBconnected())
        return "";
    const char *buf = nullptr;
    int numBytes = 0;
    if (vvMSController::instance()->isMaster())
    {
        TokenBuffer rtb;
        rtb << filePath;
        rtb << vvCommunication::instance()->getID();
        rtb << fileOwner;
        Message m(rtb);
        m.type = COVISE_MESSAGE_VRB_REQUEST_FILE;
        vv->sendVrbMessage(&m);
    }
    //wait for the file
    std::unique_ptr<Message> mymsg;
    do
    {
        std::unique_ptr<Message> msg(new Message);
        if (vvMSController::instance()->isMaster())
            vvVIVE::instance()->vrbc()->wait(msg.get());

        vvMSController::instance()->syncMessage(msg.get());
        //cache all send file messages
        if (msg->type == COVISE_MESSAGE_VRB_SEND_FILE)
        {
            m_sendFileMessages.push_back(std::move(msg));
        }
        else
        {
            vvCommunication::instance()->handleVRB(*msg);
            if(!vvPartnerList::instance()->get(fileOwner))
            {
                std::cerr << "RemoteFetch aborted: file owner disconneted" << std::endl;
                return "";
            }
        }
        
        int myID;
        std::string fn;
        auto m = m_sendFileMessages.begin();
        //find out if my file has been received
        while (m != m_sendFileMessages.end())
        {
            TokenBuffer tb(m->get());
            tb.rewind();
            tb >> myID;
            tb >> fn;
            if (fn == std::string(filePath) && myID == vvCommunication::instance()->getID())
            {
                mymsg = std::move(*m);
                m_sendFileMessages.erase(m);
                m = m_sendFileMessages.end();
            }
            else
            {
                ++m;
            }
        }
    } while (!mymsg);

    
    TokenBuffer tb(mymsg.get());
    int myID;
    std::string fn;
    tb >> myID; // this should be my ID
    tb >> fn; //this should be the requested file
    tb >> numBytes;
    if (numBytes <= 0)
    {
        return "";
    }
    buf = tb.getBinary(numBytes);
    return writeRemoteFetchedFile(filePath, buf, numBytes);

}

std::string vvFileManager::httpFetch(const std::string &url)
{
#if defined(HAVE_LIBCURL)
  std::string readBuffer;

  httpclient::curl::GET get(url);
  if (httpclient::curl::Request().httpRequest(get, readBuffer))
  {
    return writeRemoteFetchedFile(url, readBuffer.data(), readBuffer.size());
  }
  else
  {
    std::cerr << "vvFileManager::httpFetch: getting " << url << " failed" << std::endl;
  }
#else
  std::cerr << "vvFileManager::httpFetch: cannot get " << url << " - no CURL" << std::endl;
#endif
  return "";
}

size_t vvFileManager::getFileId(const std::string &url)
{
	std::string p = url;
	makeRelativeToSharedDataLink(p);
	for (size_t i = 0; i < m_sharedFiles.value().size(); i++)
	{
		if (m_sharedFiles.value()[i].first == p)
		{
			return i;
		}
	}
	return -1;
}
void vvFileManager::sendFile(TokenBuffer &tb)
{
    const char *filename;
    tb >> filename;
    int requestorsID;
    tb >> requestorsID;
    std::string validPath(filename);
    //if filenot found
    if (!fileExist(validPath))
    {
        //search file under sharedDataPath
        makeRelativeToSharedDataLink(validPath);
        validPath = m_sharedDataLink + validPath;
    }
	TokenBuffer rtb;
	rtb << requestorsID;
	rtb << filename;

	serializeFile(validPath, rtb);
	vvCommunication::instance()->send(rtb, COVISE_MESSAGE_VRB_SEND_FILE);

}
std::string vvFileManager::getFileName(const std::string &fileName)
{
    fs::path p(fileName);
    return p.filename().string();
    std::string name;
    for (size_t i = fileName.length() - 1; i > 0; --i)
    {
        if (fileName[i] == '/' || fileName[i] == '\\')
        {
            return name;
        }
        name.insert(name.begin(), fileName[i]);
    }
    return fileName;
}
std::string vvFileManager::cutFileName(const std::string &fileName)
{
    std::string name = fileName;
    for (size_t i = fileName.length() - 1; i > 0; --i)
    {
        name.pop_back();
        if (fileName[i] == '/' || fileName[i] == '\\')
        {
            return name;
        }

    }
    return "";
}
std::string vvFileManager::reduceToAlphanumeric(const std::string &str)
{
    std::locale C("C");
    std::string red;
    for (auto c: str)
    {
        // reduce to ASCII before feeding to isalnum in order to cope with invalid streams
        if (!(c & 0x80) && std::isalnum(c, C))
            red.push_back(c);
    }
    return red;
}

std::string vvFileManager::writeRemoteFetchedFile(const std::string& filePath, const char* content, int size)
{
    createRemoteFetchDir();
    fs::path path{filePath};
    auto fileName = path.filename().string();
    auto ms = vvMSController::instance();
    std::string p = remoteFetchPath;
    if (remoteFetchPathShared && ms->isSlave()) //if shared, write a shared file on master and tmp files on slaves to counter slowly updating network filesystems
        p = remoteFetchPathTmp ;
    p += "/" + getRemoteFetchHashPrefix(filePath, remoteFetchHashPrefix);
    try
    {
        if (!fs::exists(p))
            fs::create_directories(p);
    }
    catch (fs::filesystem_error)
    {
        fs::create_directories(p);
    }
    p += path.filename().string();

    if ((size > 0) && !fileExist(p))
    {
#ifndef _WIN32
        int fd = open(p.c_str(), O_RDWR | O_CREAT, 0777);
#else
        int fd = open(p.c_str(), O_RDWR | O_CREAT | O_BINARY, 0777);
#endif
        if (fd != -1)
        {
            if (int wroteBytes = write(fd, content, size) != size)
            {
                //warn("remoteFetch: temp file write error\n");
                cerr << fileName << " writing file error: " << "wrote bytes = " << wroteBytes << " received bytes = " << size << endl;
                p = "";
            }
            close(fd);
        }
        else
        {
            cerr << "opening file " << p << " failed";
            p = "";
        }
    }
    return p;
}
int vvFileManager::guessFileOwner(const std::string& fileName)
{

	int fileOwner = -1;
	int bestmatch = 0;
	for (auto p : m_sharedFiles.value())
	{
		int match = 0;
		for (size_t i = 0; i < std::min(p.first.length(), fileName.length()); i++)
		{
			if (p.first[i] == fileName[i])
			{
				++match;
			}
		}
		if (match > bestmatch)
		{
			bestmatch = match;
			fileOwner = p.second;
		}
	}
	return fileOwner;
}
bool vvFileManager::serializeFile(const std::string& fileName, covise::TokenBuffer& tb)
{
	struct stat statbuf;


	if (stat(fileName.c_str(), &statbuf) < 0)
	{
		tb << 0;
		return false;
	}
#ifdef _WIN32
	int fdesc = open(fileName.c_str(), O_RDONLY | O_BINARY);
#else
	int fdesc = open(fileName.c_str(), O_RDONLY);
#endif
	if (fdesc > 0)
	{
		tb << (int)statbuf.st_size;
		char* buf = (char*)tb.allocBinary(statbuf.st_size);
		int n = read(fdesc, buf, statbuf.st_size);
		if (n == -1)
		{
			cerr << "vvCommunication::handleVRB: read failed: " << strerror(errno) << endl;
		}
		close(fdesc);
	}
	else
	{
		cerr << " file access error, could not open " << fileName << endl;
		tb << 0;
		return false;
	}
	return true;
}
std::string vvFileManager::cutStringAt(const std::string &s, char delimiter)
{
	std::string r;
	auto it = s.begin();
	while (it != s.end() && *it != delimiter)
	{
		r.push_back(*it);
		++it;
	}
	return r;
}

std::string vvFileManager::resolveEnvs(const std::string& s)
{
	std::string stringWithoutEnvs;
#ifdef WIN32	
	char delimiter = '%';

#else 
	char delimiter = '$';
	char delimiter2 = '/';
#endif
	auto it = s.begin();
	auto lastIt = s.begin();
	while (it != s.end())
	{
		if (*it == delimiter)
		{
			stringWithoutEnvs.append(lastIt, it);
			++it;
			std::string env;
#ifdef WIN32
			while (*it != delimiter)
#else		
			while(*it != delimiter2)
#endif
			{

				env.push_back(*it);
				++it;
			}

			auto e = getenv(env.c_str());
            if(!e){
                std::cerr << "can not resolve environment variable " << env << "!" << std::endl;
                return "";
            }

            env = e;
            env = cutStringAt(env, ';');
#ifndef WIN32
			env.push_back(delimiter2);
#endif // !WIN32
			stringWithoutEnvs += env;
			++it;
			lastIt = it;
		}
		++it;
	}
	stringWithoutEnvs.append(lastIt, s.end());
	return stringWithoutEnvs;
}

const std::string &vvFileManager::getMainFile() const
{
    return m_mainFile;
}
}
