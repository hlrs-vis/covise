/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRFileManager.h"
#include <config/CoviseConfig.h>
#include "coHud.h"
#include <cassert>
#include <cstring>
#include <cctype>

#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <osgText/Font>
#include <util/unixcompat.h>
#include <util/coFileUtil.h>
#include <util/coHashIter.h>
#include "OpenCOVER.h"
#include "VRSceneGraph.h"
#include "VRViewer.h"
#include "coVRMSController.h"
#include "coVRPluginSupport.h"
#include "coVRPluginList.h"
#include "coVRCommunication.h"
#include "coTabletUI.h"
#include "coVRIOReader.h"
#include "VRRegisterSceneGraph.h"
#include "coVRConfig.h"
#include "coVRRenderer.h"
#include <util/string_util.h>
#include "ui/Owner.h"
#include "ui/Action.h"
#include "ui/Button.h"
#include "ui/Group.h"
#include "ui/FileBrowser.h"
#include "ui/Menu.h"

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <net/message.h>
#include <vrbclient/VRBClient.h>
#include <fcntl.h>


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
namespace opencover
{

coVRFileManager *coVRFileManager::s_instance = NULL;


Url::Url(const std::string &url)
{
    if (url.empty())
        return;

    auto it = url.begin();
    // must contain scheme and must begin with an aplhabet character
    if (!isalpha(*it))
        return;

    for ( ; it != url.end(); ++it)
    {
        if (std::isalnum(*it))
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
    m_scheme = std::string(url.begin(), it);
    ++it;

#ifdef WIN32
    // probably just a drive letter, not a URL scheme
    if (it - url.begin() <= 1)
        return;
#endif

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
  LoadedFile(const Url &url, ui::Button *button=nullptr)
  : url(url)
  , button(button)
  {
      if  (button)
      {
          button->setText(shortenUrl(url));
          button->setState(true);
          button->setShared(true);
          button->setCallback([this](bool state){
              if (state)
              {
                  load();
              }
              else
              {
                  if (!unload())
                  {
                      std::cerr << "unloading " << this->url << " failed" << std::endl;
                  }
              }
          });
      }
  }

  Url url;
  std::shared_ptr<ui::Button> button;
  std::string key;
  osg::Node *node = nullptr; // these should not be ref_ptrs as the group node or node might not be part of the scenegraph yet and will then be deleted immediately after loading
  osg::Group *parent = nullptr;
  const FileHandler *handler = nullptr;
  coVRIOReader *reader = nullptr;
  coTUIFileBrowserButton *filebrowser = nullptr;
  bool loaded = false;

  osg::Node *load();

  osg::Node *reload()
  {
      const char *ck = nullptr;
      if (!key.empty())
          ck = key.c_str();

      if (handler && handler->replaceFile)
      {
          handler->replaceFile(url.str().c_str(), parent, ck);
      }
      else if (unload())
      {
          load();
      }

      return nullptr;
  }

  bool unload()
  {
      if (!loaded)
          return false;

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
      else if (node && parent)
      {
          parent->removeChild(node);
          //node.release();

          ok = true;
      }

      if (ok && button)
          button->setState(false);

      loaded = false;

      return ok;
  }
};

osg::Node *LoadedFile::load()
{
    if (loaded)
        return node;

    const char *covise_key = nullptr;
    if (!key.empty())
        covise_key = key.c_str();

    auto adjustedFileName = url.str();
    auto &fb = filebrowser;

    bool isRoot = coVRFileManager::instance()->m_loadingFile==nullptr;
    if (isRoot)
        coVRFileManager::instance()->m_loadingFile = this;

    if (handler)
    {
        osg::ref_ptr<osg::Group> fakeParent = new osg::Group;
        if (cover->debugLevel(3))
            fprintf(stderr, "coVRFileManager::loadFile(name=%s)   handler\n", url.str().c_str());
        if (handler->loadUrl)
        {
            handler->loadUrl(url, fakeParent, covise_key);
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
        coVRCommunication::instance()->setCurrentFile(adjustedFileName.c_str());
        if (fakeParent->getNumChildren() == 1)
        {
            node = fakeParent->getChild(0);
        }
        for (size_t i=0; i<fakeParent->getNumChildren(); ++i)
        {
            parent->addChild(fakeParent->getChild(i));
        }
    }
    else if (reader)
    {
        //fprintf(stderr, "coVRFileManager::loadFile(name=%s)  reader\n", fileName);
        std::string filenameToLoad = adjustedFileName;

        if (fb)
        {
            filenameToLoad = fb->getFilename(adjustedFileName);
            if (filenameToLoad == "")
                filenameToLoad = adjustedFileName;
        }

        if (reader->canLoadParts())
        {
            if (cover->debugLevel(3))
                std::cerr << "coVRFileManager::loadFile info: loading parts" << std::endl;
            coVRFileManager::IOReadOperation op;
            op.filename = filenameToLoad;
            op.reader = reader;
            op.group = parent;
            coVRFileManager::instance()->readOperations[reader->getIOHandlerName()].push_back(op);
        }
        else
        {
            if (cover->debugLevel(3))
                std::cerr << "coVRFileManager::loadFile info: loading full" << std::endl;
            reader->load(filenameToLoad, parent);
        }

        coVRCommunication::instance()->setCurrentFile(adjustedFileName.c_str());
    }
    else
    {
        //fprintf(stderr, "coVRFileManager::loadFile(name=%s)   else\n", fileName);
        //(fileName,fileTypeString);
        //obj-Objects must not be rotated
        osgDB::ReaderWriter::Options *op = new osgDB::ReaderWriter::Options();
        op->setOptionString("noRotation");

        std::string tmpFileName = adjustedFileName;
        if (fb)
        {
            tmpFileName = fb->getFilename(adjustedFileName);
        }
        node = osgDB::readNodeFile(tmpFileName.c_str(), op);
        if (node)
        {
            //OpenCOVER::instance()->databasePager->registerPagedLODs(node);
            if (node->getName() == "")
            {
                node->setName(url.str());
            }
            parent->addChild(node);
            coVRCommunication::instance()->setCurrentFile(adjustedFileName.c_str());
            VRRegisterSceneGraph::instance()->registerNode(node, parent->getName());
            node->setNodeMask(node->getNodeMask() & (~Isect::Intersection));
            if (cover->debugLevel(3))
                fprintf(stderr, "coVRFileManager::loadFile setting nodeMask of %s to %x\n", node->getName().c_str(), node->getNodeMask());
        }
        else
        {
            if (covise::coFile::exists(adjustedFileName.c_str()))
                cerr << "WARNING: Could not load file " << adjustedFileName << endl;
        }

        //VRViewer::instance()->Compile();
    }

    if (isRoot)
        coVRFileManager::instance()->m_loadingFile = nullptr;

    if (button)
        button->setState(true);

    loaded = true;

    return node;
}

// load an icon file looks in covise/share/covise/icons/$LookAndFeel or covise/share/covise/icons
// returns NULL, if nothing found
osg::Node *coVRFileManager::loadIcon(const char *filename)
{
    START("coVRFileManager::loadIcon");
    osg::Node *node = NULL;
    if (node == NULL)
    {
        const char *name = NULL;
        std::string look = coCoviseConfig::getEntry("COVER.LookAndFeel");
        if (!look.empty())
        {
            std::string fn = "share/covise/icons/osg/" + look + "/" + filename + ".osg";
            name = getName(fn.c_str());
            if (!name)
            {
                std::string fn = "share/covise/icons/" + look + "/" + filename + ".iv";
                name = getName(fn.c_str());
            }
        }
        if (name == NULL)
        {
            std::string fn = "share/covise/icons/osg/";
            fn += filename;
            fn += ".osg";
            name = getName(fn.c_str());
        }
        if (name == NULL)
        {
            std::string fn = "share/covise/icons/";
            fn += filename;
            fn += ".iv";
            name = getName(fn.c_str());
        }
        if (name == NULL)
        {
            if (cover->debugLevel(4))
                fprintf(stderr, "Did not find icon %s\n", filename);
            return NULL;
        }
        node = osgDB::readNodeFile(name);
        if (node)
            node->setName(filename);
        else
        {
            fprintf(stderr, "Error loading icon %s\n", filename);
        }
    }
    return (node);
}

// parmanently loads a texture, looks in covise/icons/$LookAndFeel or covise/icons for filename.rgb
// returns NULL, if nothing found, reuses textures that have already been loaded
osg::Texture2D *coVRFileManager::loadTexture(const char *texture)
{
    START("coVRFileManager::loadTexture");

    TextureMap::iterator it = textureList.find(texture);
    if (it != textureList.end())
    {
        if (it->second.get())
        {
            if (cover->debugLevel(4))
                std::cerr << "Reusing texture " << texture << std::endl;
            return it->second.get();
        }
    }
    const char *name = buildFileName(texture);
    if (name == NULL)
    {
        if (cover->debugLevel(2))
            std::cerr << "New texture " << texture << " - not found" << std::endl;
        return NULL;
    }
    else
    {
        if (cover->debugLevel(3))
            std::cerr << "New texture " << texture << std::endl;
    }

    osg::Texture2D *tex = new osg::Texture2D();
    osg::Image *image = osgDB::readImageFile(name);
    tex->setImage(image);
    textureList[texture] = tex;

    return tex;
}

const char *coVRFileManager::buildFileName(const char *texture)
{
    const char *name = getName(texture);
    std::string look;
    if (name == NULL)
    {
        look = coCoviseConfig::getEntry("COVER.LookAndFeel");
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

bool coVRFileManager::fileExist(const char *fileName)
{
    FILE *file;
    const char *Name = buildFileName(fileName);
    if (Name)
    {
        file = ::fopen(Name, "r");
        //delete name;
        if (file)
        {
            ::fclose(file);
            return true;
        }
    }
    return false;
}

osg::Node *coVRFileManager::loadFile(const char *fileName, coTUIFileBrowserButton *fb, osg::Group *parent, const char *covise_key)
{
    if (!fileName || strcmp(fileName, "") == 0)
    {
        return nullptr;
    }
    START("coVRFileManager::loadFile");
    fs::path p(fileName);
    if (!fs::exists(p))
    {
        cerr << "file " << p.string() << " does not exist" << endl;
        return nullptr;
    }
    std::string canonicalPath = fs::canonical(p).string();
    convertBackslash(canonicalPath);
    if (m_files.find(canonicalPath) != m_files.end())
    {
        cerr << "The File : " << fileName << " is already loaded" << endl;
        cerr << "Loading a file multiple times is currently not supported" << endl;
        return nullptr;
    }
    std::string relativePath = canonicalPath;
    cutName(relativePath);
    std::string adjustedFileName;
    std::string key;

    if (fb)
    {
        //Store filename associated with corresponding fb instance
        key = canonicalPath;
        fileFBMap[key] = fb;
    }

    Url url = Url::fromFileOrUrl(canonicalPath);
    if (!url.valid())
    {
        std::cerr << "failed to parse URL " << canonicalPath << std::endl;
        return  nullptr;
    }
    std::cerr << "Loading " << url.str() << std::endl;

    if (url.scheme() == "cover"|| url.scheme() == "opencover")
    {
        if (url.authority() == "plugin")
        {
            coVRPluginList::instance()->addPlugin(url.path().c_str()+1);
        }
        return nullptr;
    }
    else if (url.scheme() == "file")
    {
        adjustedFileName = url.path();
        if (cover->debugLevel(3))
            std::cerr << " New filename: " << adjustedFileName << std::endl;
    }
    else
    {
        adjustedFileName = url.str();
    }

    if (!parent)
        parent = cover->getObjectsRoot();
    else
    {
        parent->setNodeMask(parent->getNodeMask() & (~Isect::Intersection));
        if (cover->debugLevel(3))
            fprintf(stderr, "coVRFileManager::loadFile setting nodeMask of parent to %x\n", parent->getNodeMask());
    }

    bool isRoot = m_loadingFile==nullptr;
    ui::Button *button = nullptr;
    if (cover && isRoot)
    {
        button = new ui::Button(m_fileGroup, std::string("File_"+ relativePath));
    }
    auto fe = new LoadedFile(url, button);
    if (covise_key)
        fe->key = covise_key;
    fe->filebrowser = fb;

    OpenCOVER::instance()->hud->setText2("loading");
    OpenCOVER::instance()->hud->setText3(canonicalPath);
    OpenCOVER::instance()->hud->redraw();
    if (isRoot)
    {
        if(viewPointFile == "" && url.isLocal())
        {
            const char *ext = strchr(url.path().c_str(), '.');
            if(ext)
            {
                viewPointFile = canonicalPath;
                std::string::size_type pos = viewPointFile.find_last_of('.');
                viewPointFile = viewPointFile.substr(0,pos);
                viewPointFile+=".vwp";
            }
        }
    }

    /// read the 1st line of file and try to guess the type
    std::string fileTypeString = findFileExt(url);
    const FileHandler *handler = findFileHandler(url.path().c_str());
    coVRIOReader *reader = findIOHandler(adjustedFileName.c_str());
    if (!handler && !fileTypeString.empty())
        handler = findFileHandler(fileTypeString.c_str());
    fe->handler = handler;
    fe->reader = reader;
    fe->parent = parent;

    auto node = fe->load();

    if (isRoot)
    {
        //if file is not shared, add it to the shared filePaths list
        std::set<std::string> v = filePaths;
        if (v.insert(relativePath).second)
        {
            filePaths = v;
        }
        m_files[canonicalPath] = fe;


        if (node)
            OpenCOVER::instance()->hud->setText2("done loading");
        else
            OpenCOVER::instance()->hud->setText2("failed to load");
        OpenCOVER::instance()->hud->redraw();

        m_lastFile = fe;
        fe->filebrowser = nullptr;
    }
    else
    {
        delete fe;
    }

    this->fileFBMap.erase(key);
    //VRViewer::instance()->forceCompile();
    return node;
}

osg::Node *coVRFileManager::replaceFile(const char *fileName, coTUIFileBrowserButton *fb, osg::Group *parent, const char *covise_key)
{
    if (m_lastFile)
        m_lastFile->unload();

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

std::string coVRFileManager::findFileExt(const Url &url)
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
                if (cover->debugLevel(3))
                    cerr << "coVRFileManager::findFileExt: fgets failed" << endl;
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

void coVRFileManager::reloadFile()
{
    START("coVRFileManager::reloadFile");
    if (m_lastFile)
    {
        std::cerr << "reloading " << m_lastFile->url << std::endl;
        m_lastFile->reload();
    }
}

void coVRFileManager::unloadFile(const char *file)
{
    START("coVRFileManager::unloadFile");
    if (file)
    {
        auto it = m_files.find(file);
        if (it == m_files.end())
            return;

        auto &fe = it->second;
        if (!fe->unload())
            std::cerr << "unloading " << fe->url << " failed";
    }
    else
    {
        for (auto &fe: m_files)
        {
            if (!fe.second->unload())
                std::cerr << "unloading " << fe.second->url << " failed";
        }
    }
}

coVRFileManager *coVRFileManager::instance()
{
    if (!s_instance)
        s_instance = new coVRFileManager;
    return s_instance;
}

coVRFileManager::coVRFileManager()
    : fileHandlerList()
    , filePaths("coVRFileManager_filePaths", std::set<std::string>(), vrb::ALWAYS_SHARE)
{
    START("coVRFileManager::coVRFileManager");
    /// path for the viewpoint file: initialized by 1st param() call
    assert(!s_instance);
    getSharedDataPath();
    if (cover) {
        m_owner.reset(new ui::Owner("FileManager", cover->ui));

        auto fileOpen = new ui::FileBrowser("OpenFile", m_owner.get());
        fileOpen->setText("Open");
        fileOpen->setFilter(getFilterList());
        cover->fileMenu->add(fileOpen);
        fileOpen->setCallback([this](const std::string &file){
                loadFile(file.c_str());
        });
        filePaths.setUpdateFunction([this](void) {loadPartnerFiles(); });
        m_fileGroup = new ui::Group("LoadedFiles", m_owner.get());
        m_fileGroup->setText("Files");
        cover->fileMenu->add(m_fileGroup);

        auto fileReload = new ui::Action("ReloadFile", m_owner.get());
        cover->fileMenu->add(fileReload);
        fileReload->setText("Reload file");
        fileReload->setCallback([this](){
                reloadFile();
        });

        auto fileSave = new ui::FileBrowser("SaveFile", m_owner.get(), true);
        fileSave->setText("Save");
        fileSave->setFilter(getWriteFilterList());
        cover->fileMenu->add(fileSave);
        fileSave->setCallback([this](const std::string &file){
                if (coVRMSController::instance()->isMaster())
                VRSceneGraph::instance()->saveScenegraph(file);
        });

        cover->getUpdateManager()->add(this);
    }

    osgDB::Registry::instance()->addFileExtensionAlias("gml", "citygml");
}

coVRFileManager::~coVRFileManager()
{
    START("coVRFileManager::~coVRFileManager");
    if (cover->debugLevel(2))
        fprintf(stderr, "delete coVRFileManager\n");
    cover->getUpdateManager()->remove(this);

    s_instance = NULL;
}

//=====================================================================
//
//=====================================================================
const char *coVRFileManager::getName(const char *file)
{
    START("coVRFileManager::getName");
    static char *buf = NULL;
    static int buflen = 0;

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
        if (cover->debugLevel(3))
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
        fp = ::fopen(buf, "r");
        if (fp != NULL)
        {
            fclose(fp);
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

void coVRFileManager::cutName(std::string & fileName)
{
    boost::filesystem::path filePath(fileName);
    if (!boost::filesystem::exists(filePath))
    {
        return;
    }
    if (fileName.compare(0, m_sharedDataPath.length(), m_sharedDataPath)== 0)
    {
        fileName.erase(0, m_sharedDataPath.length());
    }
}

std::string coVRFileManager::findSharedFile(const std::string & fileName)
{
    if (fs::exists(fileName))
    {
        return fileName;
    }
    std::string path = m_sharedDataPath + fileName;
    if (fs::exists(path))
    {
        return path;
    }
    else
    {
        return remoteFetch(fileName.c_str());
    }
}

std::string coVRFileManager::getFontFile(const char *fontname)
{
    std::string fontFile = "share/covise/fonts/";
    if (fontname)
    {
        fontFile += fontname;
    }
    else
    {
        fontFile += coCoviseConfig::getEntry("value", "COVER.Font", "DroidSansFallbackFull.ttf");
    }
    fontFile = getName(fontFile.c_str());
    return fontFile;
}

osg::ref_ptr<osgText::Font> coVRFileManager::loadFont(const char *fontname)
{
    osg::ref_ptr<osgText::Font> font;
    font = osgText::readRefFontFile(getFontFile(fontname));
    if (font == NULL)
    {
        font = osgText::readRefFontFile(getFontFile(NULL));
    }
    return font;
}

int coVRFileManager::coLoadFontDefaultStyle()
{

    return 0;
}

std::string coVRFileManager::getFilterList()
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

std::string coVRFileManager::getWriteFilterList()
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


const FileHandler *coVRFileManager::getFileHandler(const char *extension)
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

const FileHandler *coVRFileManager::findFileHandler(const char *pathname)
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

        int extlen = strlen(extension);
        char *cEntry = new char[40 + extlen];
        char *lowerExt = new char[extlen + 1];
        for (size_t i = 0; i < extlen; i++)
        {
            lowerExt[i] = tolower(extension[i]);
            if (lowerExt[i] == '.')
                lowerExt[i] = '_';
        }
        lowerExt[extlen] = '\0';

        sprintf(cEntry, "COVER.FileManager.FileType:%s", lowerExt);
        string plugin = coCoviseConfig::getEntry("plugin", cEntry);
        delete[] cEntry;
        delete[] lowerExt;
        if (plugin.size() > 0)
        { // load the appropriate plugin and give it another try
            coVRPluginList::instance()->addPlugin(plugin.c_str());
            for (FileHandlerList::iterator it = fileHandlerList.begin();
                    it != fileHandlerList.end();
                    ++it)
            {
                if (!strcasecmp(extension, (*it)->extension))
                {
                    if (cover->debugLevel(2))
                        fprintf(stderr, "coVRFileManager::findFileHandler(extension=%s), using plugin %s\n", extension, plugin.c_str());
                    return *it;
                }
            }
        }
    }

    return NULL;
}

coVRIOReader *coVRFileManager::findIOHandler(const char *pathname)
{
    coVRIOReader *best = NULL;
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

int coVRFileManager::registerFileHandler(const FileHandler *handler)
{
    if (getFileHandler(handler->extension))
        return -1;

    fileHandlerList.push_back(handler);
    return 0;
}

int coVRFileManager::registerFileHandler(coVRIOReader *handler)
{
    //if(getFileHandler(handler->extension))
    //   return -1;

    if (cover->debugLevel(3))
        std::cerr << "coVRFileManager::registerFileHandler info: registering " << handler->getIOHandlerName() << std::endl;

    this->ioReaderList.push_back(handler);
    return 0;
}

int coVRFileManager::unregisterFileHandler(const FileHandler *handler)
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
                && p->replaceFile == handler->replaceFile
                && p->unloadFile == handler->unloadFile)
            {
                fileHandlerList.erase(it);
                return 0;
            }
        }
    }
    return -1;
}

int coVRFileManager::unregisterFileHandler(coVRIOReader *handler)
{
    if (handler == 0)
        return -1;

    if (cover->debugLevel(3))
        std::cerr << "coVRFileManager::unregisterFileHandler info: unregistering " << handler->getIOHandlerName() << std::endl;

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

coTUIFileBrowserButton *coVRFileManager::getMatchingFileBrowserInstance(std::string keyFileName)
{
    coTUIFileBrowserButton *fbo = NULL;
    std::map<string, coTUIFileBrowserButton *>::iterator itr = this->fileFBMap.find(keyFileName);

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

bool coVRFileManager::IsDefFBSet()
{
    if (this->mDefaultFB)
        return true;
    else
        return false;
}

void coVRFileManager::SetDefaultFB(coTUIFileBrowserButton *fb)
{
    this->mDefaultFB = fb;
}

void coVRFileManager::setViewPointFile(const std::string &file)
{
    viewPointFile = file;
}

std::string coVRFileManager::getViewPointFile()
{
    return viewPointFile;
}

bool coVRFileManager::update()
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

        coVRIOReader::IOStatus status = readOperation.reader->loadPart(readOperation.filename, readOperation.group);

        if (status == coVRIOReader::Finished)
        {
            if (cover->debugLevel(3))
                std::cerr << "coVRFileManager::update info: finished loading " << readOperation.filename << std::endl;
            readOperations[reader].pop_front();
        }
        else if (status == coVRIOReader::Failed)
        {
            if (cover->debugLevel(3))
                std::cerr << "coVRFileManager::update info: failed loading " << readOperation.filename << std::endl;
            readOperations[reader].pop_front();
        }
        else
        {
            if (cover->debugLevel(3))
                std::cerr << "coVRFileManager::update info: loading " << readOperation.filename << " (" << readOperation.reader->getIOProgress() << ")" << std::endl;
        }
    }
    return true;
}

void coVRFileManager::loadPartnerFiles()
{
    //load and unload existing files
    std::set<std::string> alreadyLoadedFiles;
    for (auto myFile : m_files)
    {
        bool found = false;
        
        auto shortPath = myFile.first;
        cutName(shortPath);
        alreadyLoadedFiles.insert(shortPath);
        for (auto theirFile : filePaths.value())
        {
            if (shortPath == theirFile || myFile.first == theirFile)
            {
                myFile.second->load();
                found = true;
                break;
            }
        }
        if (!found)
        {
            unloadFile(myFile.first.c_str());
        }
    }
    
    
    //load new partner files
    std::set<std::string> newFiles;
    std::set_difference(filePaths.value().begin(), filePaths.value().end(), alreadyLoadedFiles.begin(), alreadyLoadedFiles.end(), std::inserter(newFiles, newFiles.begin()));
    for (auto newFile : newFiles)
    {
        loadFile(findSharedFile(newFile).c_str());
    }
}

void coVRFileManager::getSharedDataPath()
{
    char *covisepath = getenv("COVISE_PATH");
#ifdef WIN32
    std::vector<std::string> p = split(covisepath, ';');
#else
    std::vector<std::string> p = split(covisepath, ':');
#endif
    for (const auto path : p)
    {
        std::string link = path + "/../sharedData";
        if (fs::exists(link))
        {
            m_sharedDataPath = fs::canonical(link).string();
            convertBackslash(m_sharedDataPath);

        }
    }
}

void coVRFileManager::convertBackslash(std::string & path)
{
    std::string convertedPath;
    for (char c : path)
    {
        if (c == '\\')
        {
            convertedPath.push_back('/');
        }
        else
        {
            convertedPath.push_back(c);
        }
    }
    path = convertedPath;
}

std::string coVRFileManager::remoteFetch(const char *filename)
{
    char *result = 0;
    const char *buf = NULL;
    int numBytes = 0;
    static int working = 0;

    if (working)
    {
        cerr << "WARNING!!! reentered remoteFetch!!!!" << endl;
        return std::string();
    }

    working = 1;

    if (strncmp(filename, "vrb://", 6) == 0)
    {
        //Request file from VRB
        std::cerr << "VRB file, needs to be requested through FileBrowser-ProtocolHandler!" << std::endl;
        coTUIFileBrowserButton *locFB = coVRFileManager::instance()->getMatchingFileBrowserInstance(string(filename));
        std::string sresult = locFB->getFilename(filename).c_str();
        char *result = new char[sresult.size() + 1];
        strcpy(result, sresult.c_str());
        working = 0;
        return std::string(result);
    }
    else if (strncmp(filename, "agtk3://", 8) == 0)
    {
        //REquest file from AG data store
        std::cerr << "AccessGrid file, needs to be requested through FileBrowser-ProtocolHandler!" << std::endl;
        coTUIFileBrowserButton *locFB = coVRFileManager::instance()->getMatchingFileBrowserInstance(string(filename));
        working = 0;
        return std::string(locFB->getFilename(filename).c_str());
    }

    if (vrbc || !coVRMSController::instance()->isMaster())
    {
        if (coVRMSController::instance()->isMaster())
        {
            TokenBuffer rtb;
            rtb << filename;
            rtb << vrbc->getID();
            Message m(rtb);
            m.type = COVISE_MESSAGE_VRB_REQUEST_FILE;
            cover->sendVrbMessage(&m);
        }
        int message = 1;
        Message *msg = new Message;
        do
        {
            if (coVRMSController::instance()->isMaster())
            {
                if (!vrbc->isConnected())
                {
                    message = 0;
                    coVRMSController::instance()->sendSlaves((char *)&message, sizeof(message));
                    break;
                }
                else
                {
                    vrbc->wait(msg);
                }
                coVRMSController::instance()->sendSlaves((char *)&message, sizeof(message));
            }
            if (coVRMSController::instance()->isMaster())
            {
                coVRMSController::instance()->sendSlaves(msg);
            }
            else
            {
                coVRMSController::instance()->readMaster((char *)&message, sizeof(message));
                if (message == 0)
                    break;
                // wait for message from master instead
                coVRMSController::instance()->readMaster(msg);
            }
            coVRCommunication::instance()->handleVRB(msg);
        } while (msg->type != COVISE_MESSAGE_VRB_SEND_FILE);

        if ((msg->data) && (msg->type == COVISE_MESSAGE_VRB_SEND_FILE))
        {
            TokenBuffer tb(msg);
            int myID;
            tb >> myID; // this should be my ID
            tb >> numBytes;
            buf = tb.getBinary(numBytes);
            if ((numBytes > 0) && (result = tempnam(0, "VR")))
            {
#ifndef _WIN32
                int fd = open(result, O_RDWR | O_CREAT, 0777);
#else
                int fd = open(result, O_RDWR | O_CREAT | O_BINARY, 0777);
#endif
                if (fd != -1)
                {
                    if (write(fd, buf, numBytes) != numBytes)
                    {
                        //warn("remoteFetch: temp file write error\n");
                        free(result);
                        result = NULL;
                    }
                    close(fd);
                }
                else
                {
                    free(result);
                    result = NULL;
                }
            }
        }
        delete msg;
    }
    std::string pathToTmpFile = cutFileName(std::string(result)) + "/" + getFileName(std::string(filename));
    fs::rename(result, pathToTmpFile);
    working = 0;
    return pathToTmpFile;
}
std::string coVRFileManager::getFileName(std::string &fileName)
{
    std::string name;
    for (size_t i = fileName.length() - 1; i > 0; --i)
    {
        if (fileName[i] == '/' || fileName[i] == '\\')
        {
            return name;
        }
        name.insert(name.begin(), fileName[i]);
    }
    cerr << "invalid file path : " << fileName << endl;
    return "";
}
std::string coVRFileManager::cutFileName(std::string &fileName)
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
    cerr << "invalid file path : " << fileName << endl;
    return "";
}
}
