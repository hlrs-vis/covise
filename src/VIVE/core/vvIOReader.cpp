/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvIOReader.h"
#include "vvFileManager.h"

#include <cmath>
#include <iostream>

using namespace vive;
vvIOReader::vvIOReader()
{
    vive::vvFileManager::instance()->registerFileHandler(this);
}

vvIOReader::~vvIOReader()
{
    vive::vvFileManager::instance()->unregisterFileHandler(this);
}

vvIOReader::IOStatus vvIOReader::loadPart(const std::string &location, vsg::Group *group)
{
    (void)location;
    (void)group;

    if (canLoadParts())
        std::cerr << "vvIOReader::loadPart err: handler '" << getIOHandlerName() << "' claims to support chunked reading but does not implement loadPart" << std::endl;
    else
        std::cerr << "vvIOReader::loadPart err: internal error - loadPart called for handler not supporting chunked reads" << std::endl;

    return Failed;
}

const std::list<std::string> &vvIOReader::getSupportedReadMimeTypes() const
{
    return this->supportedReadFileTypes;
}

const std::list<std::string> &vvIOReader::getSupportedReadFileExtensions() const
{
    return this->supportedReadFileExtensions;
}

vsg::Node *vvIOReader::getLoaded()
{
    return 0;
}

bool vvIOReader::unload(vsg::Node *node)
{
    (void)node;
    return false;
}
