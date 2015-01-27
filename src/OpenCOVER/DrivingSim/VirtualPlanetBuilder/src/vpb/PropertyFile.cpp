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

#include <vpb/PropertyFile>
#include <vpb/FileUtils>

#include <string.h>

#include <iostream>
#include <sstream>
#include <string>

using namespace vpb;

namespace vpb
{

struct FileProxy
{
    typedef unsigned int offset_t;

    FileProxy(const std::string &filename)
        : _fileID(0)
        , _requiresSync(false)
    {
        if (vpb::access(filename.c_str(), F_OK) == 0)
        {
            _fileID = vpb::open(filename.c_str(), O_RDWR);

            // osg::notify(osg::NOTICE)<<"Opened existing file "<<filename<<" _fileID = "<<_fileID<<std::endl;
        }
        else
        {
            _requiresSync = true;

            FILE *file = vpb::fopen(filename.c_str(), "w+");

            vpb::fclose(file);

            _fileID = vpb::open(filename.c_str(), O_RDWR);
#if _WIN32
            vpb::fchmod(_fileID, S_IREAD | S_IWRITE | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
#else
            vpb::fchmod(_fileID, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
#endif
            vpb::fsync();

            // osg::notify(osg::NOTICE)<<"Opened new file "<<filename<<" _fileID = "<<_fileID<<std::endl;
        }
    }

    ~FileProxy()
    {
        // osg::notify(osg::NOTICE)<<"Closing _fileID = "<<_fileID<<std::endl;
        if (_fileID)
        {
            if (_requiresSync)
                vpb::fsync();

            vpb::close(_fileID);
        }
    }

    offset_t lseek(offset_t __offset, int __whence)
    {
        // osg::notify(osg::NOTICE)<<"lseek("<<_fileID<<", "<<__offset<<", "<<__whence<<")"<<std::endl;
        return vpb::lseek(_fileID, __offset, __whence);
    }

    int lockf(int __cmd, offset_t __len)
    {
        // osg::notify(osg::NOTICE)<<"lockf("<<_fileID<<", "<<__cmd<<", "<<__len<<")"<<std::endl;
        return vpb::lockf(_fileID, __cmd, __len);
    }

    ssize_t read(void *__buf, size_t __nbytes)
    {
        // osg::notify(osg::NOTICE)<<"read("<<_fileID<<", "<<__buf<<", "<<__nbytes<<")"<<std::endl;
        return vpb::read(_fileID, __buf, __nbytes);
    }

    ssize_t write(__const void *__buf, size_t __n)
    {
        // osg::notify(osg::NOTICE)<<"write("<<_fileID<<", "<<__buf<<", "<<__n<<")"<<std::endl;
        _requiresSync = true;
        return vpb::write(_fileID, __buf, __n);
    }

    int ftruncate(offset_t __length)
    {
        // osg::notify(osg::NOTICE)<<"ftruncate("<<_fileID<<", "<<__length<<")"<<std::endl;
        _requiresSync = true;
        return vpb::ftruncate(_fileID, __length);
    }

    int fsync()
    {
        _requiresSync = false;
        return vpb::fsync(_fileID);
    }

    int _fileID;
    bool _requiresSync;
};
}

bool vpb::Parameter::getString(std::string &str)
{
    switch (_type)
    {
    case (BOOL_PARAMETER):
    {
        str = *_value._bool ? "True" : "False";
        break;
    }
    case (FLOAT_PARAMETER):
    {
        std::stringstream sstr;
        sstr << *_value._float;
        str = sstr.str();
        break;
    }
    case (DOUBLE_PARAMETER):
    {
        std::stringstream sstr;
        sstr << *_value._double;
        str = sstr.str();
        break;
    }
    case (INT_PARAMETER):
    {
        std::stringstream sstr;
        sstr << *_value._int;
        str = sstr.str();
        break;
    }
    case (UNSIGNED_INT_PARAMETER):
    {
        std::stringstream sstr;
        sstr << *_value._uint;
        str = sstr.str();
        break;
    }
    case (STRING_PARAMETER):
    {
        str = *(_value._string);
        break;
    }
    }

    return true;
}

PropertyFile::PropertyFile(const std::string &filename)
    : _fileName(filename)
    , _syncCount(0)
    , _propertiesModified(false)
    , _previousSize(0)
    , _previousData(0)
    , _currentSize(0)
    , _currentData(0)
{
    // make sure the file exists.
    // FileProxy file(filename);
}

PropertyFile::~PropertyFile()
{
    if (_previousData)
        delete[] _previousData;
    if (_currentData)
        delete[] _currentData;
}

void PropertyFile::setProperty(const std::string &property, Parameter value)
{
    std::string originalValue = _propertyMap[property];
    value.getString(_propertyMap[property]);

    if (_propertyMap[property] != originalValue)
        _propertiesModified = true;
}

bool PropertyFile::getProperty(const std::string &property, Parameter value) const
{
    PropertyMap::const_iterator itr = _propertyMap.find(property);
    if (itr != _propertyMap.end())
    {
        std::string field = itr->second;
        if (value.valid(field.c_str()))
        {
            value.assign(field.c_str());
            return true;
        }
    }
    return false;
}

bool PropertyFile::read()
{
    char *data = 0;
    {
        FileProxy file(_fileName);

#if 0
        int status = 0;
        file.lseek(0, SEEK_SET);

        status = file.lockf(F_LOCK, 0);
        if (status!=0) perror("read: lock error");
#endif

        std::swap(_currentSize, _previousSize);
        std::swap(_currentData, _previousData);

        file.lseek(0, SEEK_SET);
        int size = file.lseek(0, SEEK_END);

        if (_currentSize != size)
        {
            if (_currentData)
                delete[] _currentData;
            _currentData = new char[size];
        }

        data = _currentData;

#if 1
        file.lseek(0, SEEK_SET);
#endif

        _currentSize = file.read(data, size);

// osg::notify(osg::NOTICE)<<"ProperyFile::read() filename= "<<_fileName<<" size= "<<size<<" _currentSize = "<<_currentSize<<std::endl;

#if 0
        file.lseek(0, SEEK_SET);

        status = file.lockf(F_ULOCK, 0);
        if (status!=0) perror("read: file unlock error");
#endif
    }

    bool dataChanged = (_currentSize != _previousSize) || memcmp(_currentData, _previousData, _currentSize) != 0;

    //osg::notify(osg::NOTICE)<<"ProperyFile::read() filename= "<<_fileName<<" : dataChanged "<<dataChanged<<std::endl;

    if (dataChanged)
    {

        _propertyMap.clear();

        char *end = data + _currentSize;
        char *curr = data;

        while (curr < end)
        {
            // skip over proceeding spaces
            while (*curr == ' ' && curr < end)
                ++curr;

            char *end_of_line = curr;
            while (end_of_line < end && (*end_of_line != '\n'))
                ++end_of_line;

            // wipe trailing spaces/control characters
            char *back = end_of_line - 1;
            while (back > curr && *back <= ' ')
                --back;
            ++back;

            char *colon = strchr(curr, ':');
            if (colon)
            {
                char *endName = colon - 1;
                while (endName > curr && *endName == ' ')
                    --endName;

                char *startValue = colon + 1;
                while (startValue < end && *startValue <= ' ')
                    ++startValue;

                if (startValue < end_of_line)
                {
                    _propertyMap[std::string(curr, endName + 1)] = std::string(startValue, end_of_line);
                }
                else
                {
                    _propertyMap[std::string(curr, endName + 1)] = "";
                }
            }
            else
            {
                _propertyMap[""] = std::string(curr, end_of_line);
            }

            curr = end_of_line + 1;
        }

        _propertiesModified = false;

        return true;
    }
    else
    {
        return false;
    }
}

bool PropertyFile::write()
{
    if (!_propertiesModified)
    {
        return false;
    }

    FileProxy file(_fileName);

#if 0
    file.lseek(0, SEEK_SET);

    int status = file.lockf(F_LOCK, 0);
    if (status!=0)
    {
        perror("write: file lock error");
        osg::notify(osg::NOTICE)<<"  filename: "<<_fileName<<std::endl;
    }
#endif

    file.lseek(0, SEEK_SET);

    for (PropertyMap::iterator itr = _propertyMap.begin();
         itr != _propertyMap.end();
         ++itr)
    {
        file.write(itr->first.c_str(), itr->first.length());
        file.write(" : ", 3);
        file.write(itr->second.c_str(), itr->second.length());
        file.write("\n", 1);
    }

    file.ftruncate(file.lseek(0, SEEK_CUR));

#if 0
    file.lseek(0, SEEK_SET);

    file.fsync();

    status = file.lockf(F_ULOCK, 0);
    if (status!=0)
    {
        perror("write: file unlock error");
        osg::notify(osg::NOTICE)<<"  filename: "<<_fileName<<std::endl;
    }
#endif

    _propertiesModified = false;

    return true;
}

void PropertyFile::report(std::ostream &out)
{
    out << "Properties:" << std::endl;
    for (PropertyMap::iterator itr = _propertyMap.begin();
         itr != _propertyMap.end();
         ++itr)
    {
        out << itr->first << ":" << itr->second << std::endl;
    }
}
