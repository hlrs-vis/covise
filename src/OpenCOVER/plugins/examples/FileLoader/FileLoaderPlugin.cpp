/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*!
 \brief FileLoader Plugin -
        demonstrate adding support for loading a new file format
 \author Martin Aumueller <aumueller@uni-koeln.de>
         (C) ZAIK 2008

 \date

 configure OpenCOVER to load this plugin on startup,
 then start OpenCOVER with a .txt as command line argument:
 you will see a message printed on the console
 */

#include "FileLoaderPlugin.h"
#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/RenderObject.h>

static FileHandler TextHandler = {
    FileLoaderPlugin::loadUrl,
    FileLoaderPlugin::loadFile,
    FileLoaderPlugin::unload,
    "txt"
};

FileLoaderPlugin::FileLoaderPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "FileLoaderPlugin::FileLoaderPlugin\n");

    coVRFileManager::instance()->registerFileHandler(&TextHandler);
}

// this is called if the plugin is removed at runtime
FileLoaderPlugin::~FileLoaderPlugin()
{
    coVRFileManager::instance()->unregisterFileHandler(&TextHandler);
    fprintf(stderr, "FileLoaderPlugin::~FileLoaderPlugin\n");
}

int FileLoaderPlugin::loadUrl(const Url &url, osg::Group *parent, const char *ck)
{
    /* here you should add code to transform the file data into a hierarchy
    * of OpenSceneGraph nodes and add it as a child of parent,
    * parent might be NULL */
    fprintf(stderr, "FileLoaderPlugin::loadUrl(%s,%p,%s)\n", url.str().c_str(), parent, ck);
    return 0;
}

int FileLoaderPlugin::loadFile(const char *filename, osg::Group *parent, const char *ck)
{
    /* here you should add code to transform the file data into a hierarchy
    * of OpenSceneGraph nodes and add it as a child of parent,
    * parent might be NULL */
    fprintf(stderr, "FileLoaderPlugin::loadFile(%s,%p,%s)\n", filename, parent, ck);
    return 0;
}

int FileLoaderPlugin::unload(const char *filename, const char *ck)
{
    /* here you should remove your nodes from the scene graph and
    * free any allocated resources */
    fprintf(stderr, "FileLoaderPlugin::unload(%s %s)\n", filename, ck);
    return 0;
}

COVERPLUGIN(FileLoaderPlugin)
