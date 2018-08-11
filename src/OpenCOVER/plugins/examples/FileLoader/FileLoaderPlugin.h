/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FILELOADER_PLUGIN_H
#define FILELOADER_PLUGIN_H
/*!
 \brief FileLoader Plugin -
        demonstrate adding support for loading a new file format
 \author Martin Aumueller <aumueller@uni-koeln.de>
         (C) ZAIK 2008

 \date
 */

#include <cover/coVRPlugin.h>

using namespace covise;
using namespace opencover;
namespace osg
{
class Group;
}

class FileLoaderPlugin : public coVRPlugin
{
public:
    FileLoaderPlugin();
    ~FileLoaderPlugin();

    static int loadUrl(const Url &url, osg::Group *parent, const char *ck = "");
    static int loadFile(const char *filename, osg::Group *parent, const char *ck = "");
    static int replace(const char *filename, osg::Group *parent, const char *ck = "");
    static int unload(const char *filename, const char *ck = "");
};
#endif
