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
#include "coVRPluginSupport.h"
#include "coVRPluginList.h"
#include "coVRCommunication.h"
#include "coTabletUI.h"
#include "coVRIOReader.h"
#include "coTUIFileBrowser/NetHelp.h"
#include "VRRegisterSceneGraph.h"
#include "coVRConfig.h"
#include "coVRRenderer.h"
#include <util/string_util.h>
#include "ui/Button.h"
#include "ui/Group.h"
#include "ui/FileBrowser.h"

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
    if (it - str.begin() <= 1)
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
        std::cerr << "/: " << std::string(it, slash) << std::endl;
        std::cerr << "?: " << std::string(it, question) << std::endl;
        std::cerr << "#: " << std::string(it, hash) << std::endl;
        auto end = slash;
        if (hash < end)
            end = hash;
        if (question < end)
            end = question;
        m_authority = decode(std::string(it, end));
        it = end;
        if (it != url.end())
            ++it;
    }
    else
    {
        // no authority
        it = authorityBegin;
    }

    if (m_scheme == "file")
    {
        m_path = decode(std::string(it, url.end()), true);
        m_valid = true;
        return;
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

struct FileEntry: public ui::Button {

    FileEntry(ui::Group *group, int id, const Url &url)
    : ui::Button(group, std::string("File"+std::to_string(id)))
    , url(url)
    {
        setText(shortenUrl(url));
        setState(true);
        setCallback([this](bool state){
            if (state)
            {
                coVRFileManager::instance()->loadFile(this->url.str().c_str());
            }
            else
            {
                coVRFileManager::instance()->unloadFile(this->url.str().c_str());
                delete this;
            }
        });
    }

    Url url;
};

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
    START("coVRFileManager::loadFile");

    std::string adjustedFileName;
    std::string key;
    osg::Node *result = nullptr;


    if (fb)
    {
        //Store filename associated with corresponding fb instance
        key = fileName;
        fileFBMap[key] = fb;
    }

    Url url = Url::fromFileOrUrl(fileName);
    if (!url.valid())
    {
        std::cerr << "failed to parse URL " << fileName << std::endl;
        return  nullptr;
    }
    std::cerr << "Loading " << url.str() << std::endl;

    if (url.scheme() == "cover")
    {
        if (url.authority() == "plugin")
        {
            coVRPluginList::instance()->addPlugin(url.path().c_str());
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
    OpenCOVER::instance()->hud->setText2("loading");
    OpenCOVER::instance()->hud->setText3(fileName);
    OpenCOVER::instance()->hud->redraw();
    /// read the 1st line of file and try to guess the type
    std::string fileTypeString = findFileExt(url);
    if(viewPointFile == "" && url.isLocal())
    {
        const char *ext = strchr(url.path().c_str(), '.');
        if(ext)
        {
            viewPointFile = fileName;
            std::string::size_type pos = viewPointFile.find_last_of('.');
            viewPointFile = viewPointFile.substr(0,pos);
            viewPointFile+=".vwp";
        }
    }

    new FileEntry(m_fileGroup, m_loadCount++, url);

    const FileHandler *handler = findFileHandler(url.path().c_str());
    if (!handler && !fileTypeString.empty())
        handler = findFileHandler(fileTypeString.c_str());
    coVRIOReader *reader = findIOHandler(adjustedFileName.c_str());

    delete[] lastCovise_key;
    lastFileName.clear();
    lastCovise_key = NULL;
    if (handler)
    {
        osg::ref_ptr<osg::Group> fakeParent = new osg::Group;
        if (cover->debugLevel(3))
            fprintf(stderr, "coVRFileManager::loadFile(name=%s)   handler\n", fileName);
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

                if (handler->loadFile)
                    handler->loadFile(tmpFileName.c_str(), fakeParent, covise_key);
            }
            else
            {
                if (handler->loadFile)
                    handler->loadFile(adjustedFileName.c_str(), fakeParent, covise_key);
            }
        }
        lastFileName = adjustedFileName;
        lastCovise_key = new char[strlen(covise_key) + 1];
        strcpy(lastCovise_key, covise_key);
        coVRCommunication::instance()->setCurrentFile(adjustedFileName.c_str());
        if (fakeParent->getNumChildren() == 1)
        {
            result = fakeParent->getChild(0);
        }
        for (size_t i=0; i<fakeParent->getNumChildren(); ++i)
        {
            parent->addChild(fakeParent->getChild(i));
        }
        OpenCOVER::instance()->hud->setText2("done loading");
        OpenCOVER::instance()->hud->redraw();
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
            IOReadOperation op;
            op.filename = filenameToLoad;
            op.reader = reader;
            op.group = parent;
            this->readOperations[reader->getIOHandlerName()].push_back(op);
        }
        else
        {
            if (cover->debugLevel(3))
                std::cerr << "coVRFileManager::loadFile info: loading full" << std::endl;
            reader->load(filenameToLoad, parent);
        }

        lastFileName = adjustedFileName;
        lastCovise_key = new char[strlen(covise_key) + 1];
        strcpy(lastCovise_key, covise_key);
        coVRCommunication::instance()->setCurrentFile(adjustedFileName.c_str());
        OpenCOVER::instance()->hud->setText2("done loading");
        OpenCOVER::instance()->hud->redraw();
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
        osg::Node *node = osgDB::readNodeFile(tmpFileName.c_str(), op);
        if (node)
        {
            //OpenCOVER::instance()->databasePager->registerPagedLODs(node);
            if (node->getName() == "")
            {
                node->setName(fileName);
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
                cerr << "WARNING: Could not load file " << adjustedFileName << ": no handler for " << fileTypeString << endl;
        }
        if (node)
        {
            lastFileName = adjustedFileName;
            lastCovise_key = new char[strlen(covise_key) + 1];
            strcpy(lastCovise_key, covise_key);
            OpenCOVER::instance()->hud->setText2("done loading");
        }
        else
        {
            OpenCOVER::instance()->hud->setText2("failed to load");
        }
        OpenCOVER::instance()->hud->redraw();
        lastNode = node;
        this->fileFBMap.erase(key);
        
        //VRViewer::instance()->Compile();
        return node;
    }

    this->fileFBMap.erase(key);
    //VRViewer::instance()->forceCompile();
    return result;
}

osg::Node *coVRFileManager::replaceFile(const char *fileName, coTUIFileBrowserButton *fb, osg::Group *parent, const char *covise_key)
{
    START("coVRFileManager::replaceFile");
    std::string key;
    std::string adjustedFileName;

    if (fb)
    {
        //Store filename associated with corresponding fb instance
        key = fileName;
        fileFBMap[key] = fb;
    }

    Url url = Url::fromFileOrUrl(fileName);
    if (!url.valid())
    {
        std::cerr << "coVRFileManager::replaceFile: failed to parse URL " << fileName << std::endl;
        return nullptr;
    }

    if (url.scheme() == "file")
    {
        adjustedFileName = url.path();
        if (cover->debugLevel(3))
            std::cerr << " New filename: " << adjustedFileName << std::endl;
    }
    else
    {
        adjustedFileName = url.str();
    }

    const FileHandler *oldHandler = NULL;
    if (!lastFileName.empty())
    {
        oldHandler = findFileHandler(lastFileName.c_str());
    }

    if (adjustedFileName.empty())
    {
        if (oldHandler && oldHandler->unloadFile)
            oldHandler->unloadFile(lastFileName.c_str(), lastCovise_key);
        else if (lastNode)
        {
            while (lastNode->getNumParents() > 0)
            {
                if (!parent)
                    parent = lastNode->getParent(0);
                lastNode->getParent(0)->removeChild(lastNode);
            }
        }
        lastFileName.clear();
        lastCovise_key = NULL;
    }
    else
    {
        if (!parent)
        {
            parent = cover->getObjectsRoot();
        }
        const FileHandler *handler = findFileHandler(adjustedFileName.c_str());
        if (handler)
        {
            if (handler == oldHandler && handler->replaceFile)
            {
                handler->replaceFile(adjustedFileName.c_str(), parent, covise_key);
            }
            else
            {
                if (oldHandler && oldHandler->unloadFile)
                    oldHandler->unloadFile(lastFileName.c_str(), lastCovise_key);
                handler->loadFile(adjustedFileName.c_str(), parent, covise_key);
            }
            lastFileName = adjustedFileName;
            lastCovise_key = new char[strlen(covise_key) + 1];
            strcpy(lastCovise_key, covise_key);

            // TODO: CHECK!
            // Is this really intended, to unload the new scene
            // after it has been loaded a few lines before?
            // Furthermore causes an access violation in memory
            // asumption it belongs to above if-stament's else branch
            // however there is already an unload-Statement
            // handler->unloadFile(lastFileName);
        }
        else
        {
            if (oldHandler)
            {
                oldHandler->unloadFile(lastFileName.c_str(), lastCovise_key);
            }
            delete[] lastCovise_key;
            lastCovise_key = nullptr;
            lastFileName.clear();
            lastCovise_key = NULL;

            osg::Node *node = osgDB::readNodeFile(adjustedFileName);
            if (node)
            {
                parent->addChild(node);
                lastNode = node;
            }
            else
            {
                cerr << "Could not load file " << adjustedFileName << endl;
            }

            // Store filename of new scene as lastFileName
            lastFileName = adjustedFileName;
            lastCovise_key = new char[strlen(covise_key) + 1];
            strcpy(lastCovise_key, covise_key);

            return node;
        }
    }

    this->fileFBMap.erase(key);
    return NULL;
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
    if (!lastFileName.empty() && lastCovise_key)
    {
        const FileHandler *handler = findFileHandler(lastFileName.c_str());
        if (!handler)
            return;

        if (handler->replaceFile)
            handler->replaceFile(lastFileName.c_str(), NULL, lastCovise_key);
        else if (handler->loadFile && handler->unloadFile)
        {
            handler->unloadFile(lastFileName.c_str(), lastCovise_key);
            handler->loadFile(lastFileName.c_str(), NULL, lastCovise_key);
        }
    }
}

void coVRFileManager::unloadFile(const char *file)
{
    bool lastFile = false;
    if (!file)
    {
        file = lastFileName.c_str();
        lastFile = true;
    }
    START("coVRFileManager::unloadFile");
    if (file)
    {
        const FileHandler *handler = findFileHandler(file);
        if (handler && handler->unloadFile)
            handler->unloadFile(file, lastFile ? lastCovise_key : nullptr);

        if (lastFile)
        {
            lastFileName.clear();
            delete[] lastCovise_key;
            lastCovise_key = nullptr;
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
    : ui::Owner("FileManager", cover->ui)
    , fileHandlerList()
{
    assert(!s_instance);

    m_fileOpen = new ui::FileBrowser("OpenFile", this);
    m_fileOpen->setText("Open");
    cover->fileMenu->add(m_fileOpen);

    m_fileGroup = new ui::Group("LoadedFiles", this);
    m_fileGroup->setText("Loaded files");
    cover->fileMenu->add(m_fileGroup);

    START("coVRFileManager::coVRFileManager");
    /// path for the viewpoint file: initialized by 1st param() call

    lastFileName.clear();
    lastCovise_key = NULL;
    if (cover != NULL)
        cover->getUpdateManager()->add(this);

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
    FILE *fp;
    static char *buf = NULL;
    static int buflen;

    if (file == NULL)
        return NULL;
    else if (file[0] == '\0')
        return NULL;

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
#ifdef _WIN32
        dirname = strtok(NULL, ";");
#else
        dirname = strtok(NULL, ":");
#endif
    }
    delete[] coPath;
    sprintf(buf, "%s", file);
    fp = ::fopen(buf, "r");
    if (fp != NULL)
    {
        fclose(fp);
        return buf;
    }
    buf[0] = '\0';
    return NULL;
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
    extensions += "*.osg;";
    extensions += "*.osgt;";
    extensions += "*.osgb;";
    extensions += "*.osgx;";
    extensions += "*.obj;";
    extensions += "*.stl;";
    extensions += "*.ply;";
    extensions += "*.ive;";
    extensions += "*.iv;";
    extensions += "*.dxf;";
    extensions += "*.3ds;";
    extensions += "*.flt;";
    extensions += "*.dae;";
    extensions += "*.stl;";
    extensions += "*.md2;";
    extensions += "*.geo;";
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

}
