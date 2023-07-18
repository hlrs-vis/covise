/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _GLTF_PLUGIN_H
#define _GLTF_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2023 HLRS  **
 **                                                                          **
 ** Description: GLTF Plugin (reads GLTF and GLB files)                              **
 **                                                                          **
 **                                                                          **
 ** Author: U.Woessner		                                                **
 **                                                                          **
 ** History:  								                                **
 ** JULY-23  v1	    				       		                            **
 **                                                                          **
 **                                                                          **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

class PLUGINEXPORT ListenerCover;
class PLUGINEXPORT ViewerOsg;
class VrmlScene;
class PLUGINEXPORT SystemCover;
#include <osg/Group>
#include <osg/Matrix>
#include <osg/Material>
#include "GLTFReader.h"

#include <osgDB/FileNameUtils>
#include <osgDB/Registry>

#define TINYGLTF_ENABLE_DRACO
#include "tiny_gltf.h"

class PLUGINEXPORT GLTFPlugin : public coVRPlugin
{
    friend class ListenerCover;
    friend class SystemCover;
    friend class ViewerOsg;

public:
    static GLTFPlugin *plugin;
    mutable GLTFReader::TextureCache _cache;

    GLTFPlugin();
    virtual ~GLTFPlugin();

    static int loadGLTF(const char *filename, osg::Group *loadParent, const char *ck = "");
    static int unloadGLTF(const char *filename, const char *ck = "");

    int loadFile(const std::string& fileName, osg::Group* parent);
    int unloadFile(const std::string& fileName);

    osgDB::ReaderWriter::ReadResult readNode(const std::string& location, const osgDB::Options* options);
    osgDB::ReaderWriter::ReadResult readNode(std::istream& inputStream, const osgDB::Options* options);

    // this will be called in PreFrame
    virtual void preFrame();
    virtual bool init();

    osg::ref_ptr<osg::MatrixTransform> GLTFRoot;

private:
};

#endif
