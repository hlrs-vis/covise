/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "FileReference.h"

FileReference::FileReference(const char *file)
    : osg::Referenced()
    , filename(file)
    , status(false)
{
}
FileReference::FileReference()
    : status(false)
{
}

void FileReference::setFileStatus(bool s)
{
    status = s;
}

void FileReference::setName(const char *name)
{
    f_name = name;
}

void FileReference::setId(const char *id)
{
    f_id = id;
}

std::string FileReference::getName()
{
    return f_name;
}

std::string FileReference::getId()
{
    return f_id;
}

bool FileReference::getFileStatus()
{
    return status;
}

void FileReference::addFilename(const char *file, const char *filetype)
{
    filemap[filetype] = file;
}

void FileReference::addUserValue(const char *title, const char *value)
{
    UserValues[title] = value;
}

std::string FileReference::getUserValue(const char *title)
{
    return UserValues[title];
}

std::string FileReference::getFilename(const char *filetype)
{
    std::string dest = filename;

    if (filetype != NULL)
    {
        dest = filemap[filetype];
    }
    // replace %20 with spaces

    std::string::size_type pos;
    while ((pos = dest.find("%20")) < dest.length())
    {
        dest.erase(pos, 2);
        dest[pos] = ' ';
    }
    return dest;
}
