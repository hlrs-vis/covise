/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRFileManager.h>

#include <PluginUtil/FileReference.h>
#include <PluginUtil/SimReference.h>
#include "LoadFileVisitor.h"

using namespace opencover;

LoadFileVisitor::LoadFileVisitor()
    : osg::NodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN)
{
}
LoadFileVisitor::~LoadFileVisitor()
{
}

void LoadFileVisitor::apply(osg::Group &node)
{
    osg::Referenced *data = node.getUserData();
    FileReference *fileRef;
    if ((fileRef = dynamic_cast<FileReference *>(data)))
    {
        if (!fileRef->getFileStatus())
        {
            coVRFileManager::instance()->loadFile(fileRef->getFilename().c_str(), NULL, &node, "");
            fileRef->setFileStatus(true);
        }
    }
    traverse(node);
}
//-----------------------------------------------------------------------------------------------
LoadFile::LoadFile()
{
}
LoadFile::~LoadFile()
{
}

void LoadFile::load(osg::Group *node)
{
    osg::Referenced *data = node->getUserData();
    FileReference *fileRef;
    if ((fileRef = dynamic_cast<FileReference *>(data)))
    {
        if (!fileRef->getFileStatus())
        {
            coVRFileManager::instance()->loadFile(fileRef->getFilename().c_str(), NULL, node, "");
            fileRef->setFileStatus(true);
        }
    }
}
