/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ASSIMP_PLUGIN_H
#define _ASSIMP_PLUGIN_H
/****************************************************************************\
 **                                                            (C)2025 ITCC  **
 **                                                                          **
 ** Description: Assimp Plugin (reads 3D model files via Assimp library)    **
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
using namespace covise;
using namespace opencover;

#include <osg/Group>
#include <osg/Matrix>
#include <osg/Material>
#include "AssimpReader.h"

#include <osgDB/FileNameUtils>
#include <osgDB/Registry>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

class PLUGINEXPORT AssimpPlugin : public coVRPlugin
{
public:
    static AssimpPlugin *plugin;
    mutable AssimpReader::TextureCache _cache;

    AssimpPlugin();
    virtual ~AssimpPlugin();

    static int loadAssimp(const char *filename, osg::Group *loadParent, const char *ck = "");
    static int unloadAssimp(const char *filename, const char *ck = "");

    int loadFile(const std::string& fileName, osg::Group* parent);
    int unloadFile(const std::string& fileName);

    osgDB::ReaderWriter::ReadResult readNode(const std::string& location, const osgDB::Options* options);

    void preFrame() override;
    bool init() override;

    osg::ref_ptr<osg::MatrixTransform> AssimpRoot;

private:
};

#endif
