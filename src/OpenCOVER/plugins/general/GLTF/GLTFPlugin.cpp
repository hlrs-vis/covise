/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
**                                                            (C)2021 HLRS  **
**                                                                          **
** Description: GLTF Plugin (read GLTF files)                                 **
**                                                                          **
**                                                                          **
** Author: U.Woessner		                                                **
**                                                                          **
** History:  								                                **
** Sep-21  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <PluginUtil/coLOD.h>
#include <config/CoviseConfig.h>
#include <cover/coVRShader.h>
#include <codecvt>
#include <string>


#include <cover/RenderObject.h>
#include <cover/VRRegisterSceneGraph.h>
#include <osg/LOD>

#include <osg/GL>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/TexGen>
#include <osg/TexEnv>
#include <osg/TexMat>
#include <osg/TexEnvCombine>
#include <osg/Texture>
#include <osg/TextureCubeMap>
#include <osg/Texture2D>
#include <osg/Geode>
#include <osg/Switch>
#include <osg/Geometry>
#include <osg/PrimitiveSet>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/CullFace>
#include <osg/BlendFunc>
#include <osg/Light>
#include <osg/LightSource>
#include <osg/Depth>
#include <osg/Fog>
#include <osg/AlphaFunc>
#include <osg/ColorMask>
#include <osgDB/ReadFile>

#include <osgUtil/SmoothingVisitor>

#include <osgUtil/Optimizer>
#include "GLTFPlugin.h"

#ifdef HAVE_OSGNV
#include <osgNVExt/RegisterCombiners>
#include <osgNVExt/CombinerInput>
#include <osgNVExt/CombinerOutput>
#include <osgNVExt/FinalCombinerInput>
#endif

#include <osgText/Font>
#include <osgText/Text>


#include <sys/types.h>
#include <string.h>

#include <boost/locale.hpp>


GLTFPlugin *GLTFPlugin::plugin = NULL;

static FileHandler handlers[] = {
    {NULL,
      GLTFPlugin::loadGLTF,
      GLTFPlugin::unloadGLTF,
      "gltf"},
      {NULL,
      GLTFPlugin::loadGLTF,
      GLTFPlugin::unloadGLTF,
      "glb"}
};



int GLTFPlugin::loadGLTF(const char* filename, osg::Group* loadParent, const char*)
{
#ifdef WIN32
	std::string utf8_filename = boost::locale::conv::to_utf<char>(filename, "ISO-8859-1"); // we hope  the system locale is Latin1
#else
	std::string utf8_filename(filename); // we hope it is utf8 already
#endif
	return plugin->loadFile(utf8_filename, loadParent);
}

int GLTFPlugin::unloadFile(const std::string& fileName)
{
	return 0;
}

int GLTFPlugin::loadFile(const std::string&fileName, osg::Group *loadParent)
{
	if (loadParent == nullptr)
	{
		loadParent = GLTFRoot;
	}
    osgDB::ReaderWriter::ReadResult res = readNode(fileName, nullptr);

	loadParent->addChild(res.takeNode());

    return 0;
}

int GLTFPlugin::unloadGLTF(const char *filename, const char *)
{
	return plugin->unloadFile(filename);
}

osgDB::ReaderWriter::ReadResult GLTFPlugin::readNode(const std::string& location, const osgDB::Options* options)
{
    std::string ext = osgDB::getFileExtension(location);

    if (ext == "gltf")
    {
        GLTFReader reader;
        reader.setTextureCache(&_cache);
        tinygltf::Model model;
        return reader.read(location, false, options);
    }
    else if (ext == "glb")
    {
        GLTFReader reader;
        reader.setTextureCache(&_cache);
        tinygltf::Model model;
        return reader.read(location, true, options);
    }
    else return osgDB::ReaderWriter::ReadResult::FILE_NOT_HANDLED;
}

//! Read from a stream:
osgDB::ReaderWriter::ReadResult GLTFPlugin::readNode(std::istream& inputStream, const osgDB::Options* options)
{
    // load entire stream into a buffer
    std::istreambuf_iterator<char> eof;
    std::string buffer(std::istreambuf_iterator<char>(inputStream), eof);

#if ENABLE_OE
    // Find referrer in the options
    URIContext context(options);
#endif // ENABLE_OE

    // Determine format by peeking the magic header:
    std::string magic(buffer, 0, 4);

    if (magic == "glTF")
    {
        // non-functional -- fix - TODO
        GLTFReader reader;
        reader.setTextureCache(&_cache);
        tinygltf::Model model;
#if ENABLE_OE
        return reader.read(context.referrer(), true, options); // binary=yes
#else
        return reader.read(nullptr, true, options);
#endif // ENABLE_OE
    }
    //else if (magic == "b3dm")
    //{
    //    B3DMReader reader;
    //    reader.setTextureCache(&_cache);
    //    return reader.read(context.referrer(), buffer, options);
    //}
    else return osgDB::ReaderWriter::ReadResult::FILE_NOT_HANDLED;
}

GLTFPlugin::GLTFPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
   
    //scaleGeometry = coCoviseConfig::isOn("COVER.Plugin.GLTF.ScaleGeometry", true);
    //lodScale = coCoviseConfig::getFloat("COVER.Plugin.GLTF.LodScale", 2000.0f);
	GLTFRoot = new osg::MatrixTransform();
	GLTFRoot->setName("GLTFRoot");
	cover->getObjectsRoot()->addChild(GLTFRoot);
    plugin = this;
}

bool GLTFPlugin::init()
{
    for(int i=0;i<2;i++)
    coVRFileManager::instance()->registerFileHandler(&handlers[i]);
    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
GLTFPlugin::~GLTFPlugin()
{
    for (int i = 0; i < 2; i++)
    coVRFileManager::instance()->unregisterFileHandler(&handlers[i]);
	cover->getObjectsRoot()->removeChild(GLTFRoot);
}

void
GLTFPlugin::preFrame()
{
}

COVERPLUGIN(GLTFPlugin)
