/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2025 ITCC  **
**                                                                          **
** Description: Assimp Plugin (read 3D model files via Assimp)             **
**                                                                          **
**                                                                          **
** Author: D.Wickeroth		                                                **
**                                                                          **
** History:  								                                **
** Dec-25  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <config/CoviseConfig.h>
#include <boost/locale.hpp>
#include <string>

#include "AssimpPlugin.h"

AssimpPlugin *AssimpPlugin::plugin = NULL;

// Static file handler array - register ALL Assimp formats
static FileHandler handlers[] = {
    // Common formats
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "obj"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "fbx"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "dae"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "3ds"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "blend"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "stl"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "ply"},

    // CAD formats
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "dxf"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "ifc"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "step"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "stp"},

    // Game and animation formats
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "bvh"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "md2"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "md3"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "md5mesh"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "mdl"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "x"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "ac"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "ms3d"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "cob"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "scn"},

    // Scene formats
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "irrmesh"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "irr"},

    // Other formats
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "lwo"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "lws"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "lxo"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "off"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "b3d"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "ase"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "smd"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "3mf"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "xgl"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "zgl"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "ogex"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "q3d"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "q3s"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "raw"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "ter"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "hmp"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "ndo"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "csm"},
    {NULL, AssimpPlugin::loadAssimp, AssimpPlugin::unloadAssimp, "vta"},
};

int AssimpPlugin::loadAssimp(const char* filename, osg::Group* loadParent, const char*)
{
#ifdef WIN32
    std::string utf8_filename = boost::locale::conv::to_utf<char>(filename, "ISO-8859-1");
#else
    std::string utf8_filename(filename);
#endif
    return plugin->loadFile(utf8_filename, loadParent);
}

int AssimpPlugin::unloadFile(const std::string& fileName)
{
    // TODO: Implement unloading logic if needed
    return 0;
}

int AssimpPlugin::loadFile(const std::string& fileName, osg::Group *loadParent)
{
    if (loadParent == nullptr)
    {
        loadParent = AssimpRoot.get();
    }

    osgDB::ReaderWriter::ReadResult res = readNode(fileName, nullptr);

    if (res.success())
    {
        loadParent->addChild(res.takeNode());
        return 0;
    }
    else
    {
        OSG_WARN << "AssimpPlugin: Failed to load " << fileName << std::endl;
        return -1;
    }
}

int AssimpPlugin::unloadAssimp(const char *filename, const char *)
{
    return plugin->unloadFile(filename);
}

osgDB::ReaderWriter::ReadResult AssimpPlugin::readNode(const std::string& location, const osgDB::Options* options)
{
    AssimpReader reader;
    reader.setTextureCache(&_cache);
    return reader.read(location, options);
}

AssimpPlugin::AssimpPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    AssimpRoot = new osg::MatrixTransform();
    AssimpRoot->setName("AssimpRoot");
    cover->getObjectsRoot()->addChild(AssimpRoot.get());
    plugin = this;
}

bool AssimpPlugin::init()
{   
    std::cout << "AssimpPlugin::init()" << std::endl;

    int numHandlers = sizeof(handlers) / sizeof(FileHandler);
    for(int i = 0; i < numHandlers; i++)
        coVRFileManager::instance()->registerFileHandler(&handlers[i]);
    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
AssimpPlugin::~AssimpPlugin()
{
    int numHandlers = sizeof(handlers) / sizeof(FileHandler);
    for (int i = 0; i < numHandlers; i++)
        coVRFileManager::instance()->unregisterFileHandler(&handlers[i]);
    cover->getObjectsRoot()->removeChild(AssimpRoot.get());
}

void AssimpPlugin::preFrame()
{
}

COVERPLUGIN(AssimpPlugin)

// OSG Plugin registration
class AssimpLoader : public osgDB::ReaderWriter
{
public:
    AssimpLoader()
    {
        // Register all supported extensions
        supportsExtension("obj", "Assimp model loader");
        supportsExtension("fbx", "Assimp model loader");
        supportsExtension("dae", "Assimp model loader");
        supportsExtension("3ds", "Assimp model loader");
        supportsExtension("blend", "Assimp model loader");
        supportsExtension("stl", "Assimp model loader");
        supportsExtension("ply", "Assimp model loader");
        supportsExtension("dxf", "Assimp model loader");
        supportsExtension("ifc", "Assimp model loader");
        supportsExtension("step", "Assimp model loader");
        supportsExtension("stp", "Assimp model loader");
        supportsExtension("bvh", "Assimp model loader");
        supportsExtension("md2", "Assimp model loader");
        supportsExtension("md3", "Assimp model loader");
        supportsExtension("md5mesh", "Assimp model loader");
        supportsExtension("mdl", "Assimp model loader");
        supportsExtension("x", "Assimp model loader");
        supportsExtension("ac", "Assimp model loader");
        supportsExtension("ms3d", "Assimp model loader");
        supportsExtension("cob", "Assimp model loader");
        supportsExtension("scn", "Assimp model loader");
        supportsExtension("irrmesh", "Assimp model loader");
        supportsExtension("irr", "Assimp model loader");
        supportsExtension("lwo", "Assimp model loader");
        supportsExtension("lws", "Assimp model loader");
        supportsExtension("lxo", "Assimp model loader");
        supportsExtension("off", "Assimp model loader");
        supportsExtension("b3d", "Assimp model loader");
        supportsExtension("ase", "Assimp model loader");
        supportsExtension("smd", "Assimp model loader");
        supportsExtension("3mf", "Assimp model loader");
        supportsExtension("xgl", "Assimp model loader");
        supportsExtension("zgl", "Assimp model loader");
        supportsExtension("ogex", "Assimp model loader");
        supportsExtension("q3d", "Assimp model loader");
        supportsExtension("q3s", "Assimp model loader");
        supportsExtension("raw", "Assimp model loader");
        supportsExtension("ter", "Assimp model loader");
        supportsExtension("hmp", "Assimp model loader");
        supportsExtension("ndo", "Assimp model loader");
        supportsExtension("csm", "Assimp model loader");
        supportsExtension("vta", "Assimp model loader");
    }

    virtual const char* className() const { return "AssimpLoader"; }

    virtual ReadResult readObject(const std::string& filename, const osgDB::ReaderWriter::Options* options) const
    {
        return readNode(filename, options);
    }

    virtual ReadResult readNode(const std::string& fileName, const osgDB::ReaderWriter::Options* options) const
    {
        if (AssimpPlugin::plugin)
            return AssimpPlugin::plugin->readNode(fileName, options);
        return ReadResult::FILE_NOT_HANDLED;
    }
};

REGISTER_OSGPLUGIN(assimp, AssimpLoader);
