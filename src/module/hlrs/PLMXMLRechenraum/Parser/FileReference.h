/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FILE_REFERENCE
#define FILE_REFERENCE

#include <osg/Referenced>
#include <string>
#include <map>
#include <cstring>

class FileReference : public osg::Referenced
{
public:
    FileReference(const char *file);
    FileReference();

    void setFileStatus(bool status);
    bool getFileStatus();
    std::string getFilename(const char *filetype = NULL);

    void addFilename(const char *file, const char *filetype);

    void addUserValue(const char *value, const char *title);

    std::string getUserValue(const char *title);

private:
    class compare
    {

    public:
        bool operator()(std::string s1, std::string s2) const
        {
#ifdef WIN32
            return stricmp(s1.c_str(), s2.c_str()) < 0;
#else
            return strcasecmp(s1.c_str(), s2.c_str()) < 0;
#endif
        }
    };

    std::string filename;
    bool status;
    std::map<std::string, std::string, compare> filemap;
    std::map<std::string, std::string> UserValues;
    std::string value;
    std::string title;
};
#endif
