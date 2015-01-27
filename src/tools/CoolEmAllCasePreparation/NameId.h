/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef NAME_ID_H
#define NAME_ID_H

#include <osg/Referenced>
#include <string>
#include <map>
#include <cstring>

class NameId : public osg::Referenced
{

    std::string m_name;
    std::string m_id;
    std::string m_filename;
    std::map<std::string, std::string> UserValues;
    std::string title;
    std::string value;

public:
    void setName(const char *name);

    void setId(const char *id);

    std::string getName();

    std::string getId();

    std::string makeFileName();

    void addUserValue(const char *value, const char *title);

    std::string getUserValue(const char *title);
};

#endif
