/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRIOReader.h"
#include "vvFileManager.h"

#include <cmath>
#include <iostream>

using namespace vive;
coVRIOReader::coVRIOReader()
{
    vive::vvFileManager::instance()->registerFileHandler(this);
}

coVRIOReader::~coVRIOReader()
{
    vive::vvFileManager::instance()->unregisterFileHandler(this);
}

coVRIOReader::IOStatus coVRIOReader::loadPart(const std::string &location, vsg::Group *group)
{
    (void)location;
    (void)group;

    if (canLoadParts())
        std::cerr << "coVRIOReader::loadPart err: handler '" << getIOHandlerName() << "' claims to support chunked reading but does not implement loadPart" << std::endl;
    else
        std::cerr << "coVRIOReader::loadPart err: internal error - loadPart called for handler not supporting chunked reads" << std::endl;

    return Failed;
}

const std::list<std::string> &coVRIOReader::getSupportedReadMimeTypes() const
{
    return this->supportedReadFileTypes;
}

const std::list<std::string> &coVRIOReader::getSupportedReadFileExtensions() const
{
    return this->supportedReadFileExtensions;
}

vsg::Node *coVRIOReader::getLoaded()
{
    return 0;
}

bool coVRIOReader::unload(vsg::Node *node)
{
    (void)node;
    return false;
}
