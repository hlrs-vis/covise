/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* -*-c++-*- VirtualPlanetBuilder - Copyright (C) 1998-2009 Robert Osfield
 *
 * This library is open source and may be redistributed and/or modified under
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * OpenSceneGraph Public License for more details.
*/

#include <vpb/FilePathManager>
#include <OpenThreads/ScopedLock>

using namespace vpb;

FilePathManager::FilePathManager()
{
}

FilePathManager::~FilePathManager()
{
}

osg::ref_ptr<vpb::FilePathManager> &FilePathManager::instance()
{
    static osg::ref_ptr<FilePathManager> s_FilePathManager = new FilePathManager;
    return s_FilePathManager;
}

osgDB::FileType FilePathManager::getFileType(const std::string &filename)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);
    FilePathTypeMap::iterator itr = _filePathTypeMap.find(filename);
    if (itr != _filePathTypeMap.end())
        return itr->second;

    osgDB::FileType type = osgDB::fileType(filename);
    if (type == osgDB::FILE_NOT_FOUND)
        return osgDB::FILE_NOT_FOUND;

    _filePathTypeMap[filename] = type;
    return type;
}

bool FilePathManager::checkWritePermissionAndEnsurePathAvailability(const std::string &filename)
{
    OpenThreads::ScopedLock<OpenThreads::Mutex> lock(_mutex);

    FilePathPermissionMap::iterator itr = _filePathWritePermissionMap.find(filename);
    if (itr != _filePathWritePermissionMap.end())
        return itr->second;

    std::string path = osgDB::getFilePath(filename);
    if (path.empty())
        path = ".";

    itr = _filePathWritePermissionMap.find(path);
    if (itr != _filePathWritePermissionMap.end())
        return itr->second;

    if (vpb::access(filename.c_str(), W_OK) == 0)
    {
        _filePathWritePermissionMap[filename] = true;
        return true;
    }

    if (vpb::access(path.c_str(), W_OK) == 0)
    {
        _filePathWritePermissionMap[path] = true;
        return true;
    }

    int result = vpb::mkpath(path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
    if (result == 0)
    {
        _filePathWritePermissionMap[path] = true;
        return true;
    }

    _filePathWritePermissionMap[path] = false;
    return false;
}
