/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRFileManager.h"
#include <config/CoviseConfig.h>
#include "coHud.h"
#include <assert.h>
#include <string.h>

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
using namespace opencover;

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

osg::Node *coVRFileManager::getLastModelNode()
{
    START("coVRFileManager::getLastModelNode");
    return lastNode;
}

osg::Node *coVRFileManager::loadFile(const char *fileName, coTUIFileBrowserButton *fb, osg::Group *parent, const char *covise_key)
{
    START("coVRFileManager::loadFile");
   

    char *adjustedFileName = NULL;
    std::string key;
    bool allocated = false;


    if (fb)
    {
        //Store filename associated with corresponding fb instance
        key = fileName;
        fileFBMap[key] = fb;
    }

    if (strncmp(fileName, "file://", 7) == 0)
    {
        adjustedFileName = new char[sizeof(char) * strlen(fileName)];
        allocated = true;
        strncpy(adjustedFileName, strstr(fileName, "file://") + 7, strlen(fileName) - 6);
        if (cover->debugLevel(3))
            std::cerr << " New filename: " << adjustedFileName << std::endl;
    }
    else
    {
        std::string tempFN = fileName;
        std::string::size_type pos = tempFN.find("://", 0);
        if (pos != std::string::npos)
        {
            pos += 3;
            std::string::size_type end = tempFN.find("/", pos);
            std::string strIP = tempFN.substr(pos, end - pos);
            NetHelp net;
            if (strIP.compare(net.getLocalIP().toStdString()) == 0)
            {
                std::string fileLocation = tempFN.substr(end + 1, tempFN.size() - end);
                adjustedFileName = new char[sizeof(char) * (fileLocation.size() + 1)];
                strncpy(adjustedFileName, fileLocation.c_str(), fileLocation.size());
                adjustedFileName[fileLocation.size()] = '\0';
                allocated = true;
            }
            else
            {
                adjustedFileName = (char *)fileName;
            }
        }
        else
        {
            adjustedFileName = (char *)fileName;
        }
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
    char fileTypeBuf[10] = "";
    const char *fileTypeString = findFileExt(adjustedFileName);
    if(viewPointFile == "")
    {
        
        const char *ext = strchr(adjustedFileName, '.');
        if(ext)
        {
            viewPointFile = fileName;
            std::string::size_type pos = viewPointFile.find_last_of('.');
            viewPointFile = viewPointFile.substr(0,pos);
            viewPointFile+=".vwp";
        }

    }
    if (!strcmp(fileTypeString, adjustedFileName))
    {
        if (!strncmp(adjustedFileName, "http://", 7) || !strncmp(adjustedFileName, "file://", 7))
        {
            char *url = new char[strlen(fileName) + 1];
            strcpy(url, fileName);
            char *p = strchr(url, '?');
            if (p)
            {
                *p = '\0';
                p++;
            }
            p = strrchr(url, '.');
            if (p)
            {
                strncpy(fileTypeBuf, p + 1, sizeof(fileTypeBuf));
                fileTypeBuf[sizeof(fileTypeBuf) - 1] = '\0';
                fileTypeString = fileTypeBuf;
            }
            delete[] url;
        }
    }

    const FileHandler *handler = findFileHandler(adjustedFileName);
    coVRIOReader *reader = findIOHandler(adjustedFileName);

    delete[] lastFileName;
    delete[] lastCovise_key;
    lastFileName = NULL;
    lastCovise_key = NULL;
    if (handler)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "coVRFileManager::loadFile(name=%s)   handler\n", fileName);
        if (handler->loadUrl)
        {
            handler->loadUrl(adjustedFileName, parent, covise_key);
        }
        else
        {
            if (fb)
            {
                std::string tmpFileName = fb->getFilename(adjustedFileName);
                if (tmpFileName == "")
                    tmpFileName = adjustedFileName;

                if (handler->loadFile)
                    handler->loadFile(tmpFileName.c_str(), parent, covise_key);
            }
            else
            {
                if (handler->loadFile)
                    handler->loadFile(adjustedFileName, parent, covise_key);
            }
        }
        lastFileName = new char[strlen(adjustedFileName) + 1];
        strcpy(lastFileName, adjustedFileName);
        lastCovise_key = new char[strlen(covise_key) + 1];
        strcpy(lastCovise_key, covise_key);
        coVRCommunication::instance()->setCurrentFile(adjustedFileName);
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

        lastFileName = new char[strlen(adjustedFileName) + 1];
        strcpy(lastFileName, adjustedFileName);
        lastCovise_key = new char[strlen(covise_key) + 1];
        strcpy(lastCovise_key, covise_key);
        coVRCommunication::instance()->setCurrentFile(adjustedFileName);
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
            coVRCommunication::instance()->setCurrentFile(adjustedFileName);
            VRRegisterSceneGraph::instance()->registerNode(node, parent->getName());
            node->setNodeMask(node->getNodeMask() & (~Isect::Intersection));
            if (cover->debugLevel(3))
                fprintf(stderr, "coVRFileManager::loadFile setting nodeMask of %s to %x\n", node->getName().c_str(), node->getNodeMask());
        }
        else
        {
            if (covise::coFile::exists(adjustedFileName))
                cerr << "WARNING: Could not load file " << adjustedFileName << ": no handler for " << fileTypeString << endl;
        }
        if (node)
        {
            lastFileName = new char[strlen(adjustedFileName) + 1];
            strcpy(lastFileName, adjustedFileName);
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
    if (allocated)
    {
        delete[] adjustedFileName;
    }
    //VRViewer::instance()->forceCompile();
    return NULL;
}

osg::Node *coVRFileManager::replaceFile(const char *fileName, coTUIFileBrowserButton *fb, osg::Group *parent, const char *covise_key)
{
    START("coVRFileManager::replaceFile");
    std::string key;
    char *adjustedFileName = NULL;
    bool allocated = false;

    if (fb)
    {
        //Store filename associated with corresponding fb instance
        key = fileName;
        fileFBMap[key] = fb;
    }

    if (strncmp(fileName, "file://", 7) == 0)
    {
        adjustedFileName = new char[sizeof(char) * strlen(fileName)];
        allocated = true;
        strncpy(adjustedFileName, strstr(fileName, "file://") + 7, strlen(fileName) - 6);
        if (cover->debugLevel(3))
            std::cerr << " New filename: " << adjustedFileName << std::endl;
    }
    else
    {
        std::string tempFN = fileName;
        std::string::size_type pos = tempFN.find("://", 0);
        if (pos != std::string::npos)
        {
            pos += 3;
            std::string::size_type end = tempFN.find("/", pos);
            std::string strIP = tempFN.substr(pos, end - pos);
            NetHelp net;
            if (strIP.compare(net.getLocalIP().toStdString()) == 0)
            {
                std::string fileLocation = tempFN.substr(end + 1, tempFN.size() - end);
                adjustedFileName = new char[sizeof(char) * (fileLocation.size() + 1)];
                strncpy(adjustedFileName, fileLocation.c_str(), fileLocation.size());
                adjustedFileName[fileLocation.size()] = '\0';
                allocated = true;
            }
            else
            {
                adjustedFileName = (char *)fileName;
            }
        }
        else
        {
            adjustedFileName = (char *)fileName;
        }
    }

    const FileHandler *oldHandler = NULL;
    if (lastFileName)
    {
        oldHandler = findFileHandler(lastFileName);
    }

    if (adjustedFileName == NULL)
    {
        if (oldHandler && oldHandler->unloadFile)
            oldHandler->unloadFile(lastFileName, lastCovise_key);
        else if (lastNode)
        {
            while (lastNode->getNumParents() > 0)
            {
                if (!parent)
                    parent = lastNode->getParent(0);
                lastNode->getParent(0)->removeChild(lastNode);
            }
        }
        lastFileName = NULL;
        lastCovise_key = NULL;
    }
    else
    {
        if (!parent)
        {
            parent = cover->getObjectsRoot();
        }
        const FileHandler *handler = findFileHandler(adjustedFileName);
        if (handler)
        {
            if (handler == oldHandler && handler->replaceFile)
            {
                handler->replaceFile(adjustedFileName, parent, covise_key);
            }
            else
            {
                if (oldHandler && oldHandler->unloadFile)
                    oldHandler->unloadFile(lastFileName, lastCovise_key);
                handler->loadFile(adjustedFileName, parent, covise_key);
            }
            lastFileName = new char[strlen(adjustedFileName) + 1];
            lastCovise_key = new char[strlen(covise_key) + 1];
            strcpy(lastFileName, adjustedFileName);
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
                oldHandler->unloadFile(lastFileName, lastCovise_key);
            }
            delete lastFileName;
            delete lastCovise_key;
            lastFileName = NULL;
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
            if (allocated)
            {
                delete[] lastFileName;
                delete[] lastCovise_key;
            }
            lastFileName = new char[strlen(adjustedFileName) + 1];
            strcpy(lastFileName, adjustedFileName);
            lastCovise_key = new char[strlen(covise_key) + 1];
            strcpy(lastCovise_key, covise_key);

            return node;
        }
    }

    this->fileFBMap.erase(key);
    if (allocated)
    {
        delete[] adjustedFileName;
    }
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

const char *coVRFileManager::findFileExt(const char *filename)
{
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
    /* look for final "." in filename */
    const char *ext = strchr(filename, '.');
    /* no dot, assume it's just the extension */
    if (ext == NULL)
        return filename;
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
    if (lastFileName && lastCovise_key)
    {
        const FileHandler *handler = findFileHandler(lastFileName);
        if (!handler)
            return;

        if (handler->replaceFile)
            handler->replaceFile(lastFileName, NULL, lastCovise_key);
        else if (handler->loadFile && handler->unloadFile)
        {
            handler->unloadFile(lastFileName, lastCovise_key);
            handler->loadFile(lastFileName, NULL, lastCovise_key);
        }
    }
}

void coVRFileManager::unloadFile()
{
    START("coVRFileManager::unloadFile");
    if (lastFileName && lastCovise_key)
    {
        const FileHandler *handler = findFileHandler(lastFileName);
        if (handler && handler->unloadFile)
            handler->unloadFile(lastFileName, lastCovise_key);
    }
}

coVRFileManager *
coVRFileManager::instance()
{
    static coVRFileManager *singleton = NULL;
    if (!singleton)
        singleton = new coVRFileManager;
    return singleton;
}

coVRFileManager::coVRFileManager()
    : fileHandlerList()
{
    START("coVRFileManager::coVRFileManager");
    /// path for the viewpoint file: initialized by 1st param() call

    lastFileName = NULL;
    lastCovise_key = NULL;
    if (cover != NULL)
        cover->getUpdateManager()->add(this);
}

coVRFileManager::~coVRFileManager()
{
    START("coVRFileManager::~coVRFileManager");
    if (cover->debugLevel(2))
        fprintf(stderr, "delete coVRFileManager\n");
    cover->getUpdateManager()->remove(this);
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
    for (const char *p = strchr(pathname, '.'); p; p = strchr(p, '.'))
    {
        ++p;
        const char *extension = p;

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
