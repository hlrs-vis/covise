/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "NameId.h"

void NameId::setName(const char *name)
{
    m_name = name;
}

void NameId::setId(const char *id)
{
    m_id = id;
}

std::string NameId::getName()
{
    return m_name;
}

std::string NameId::getId()
{
    return m_id;
}

std::string NameId::makeFileName()
{
    m_filename = m_name + "@" + m_id + ".stl";
    return m_filename;
}

void NameId::addUserValue(const char *value, const char *title)
{
    UserValues[title] = value;
}
std::string NameId::getUserValue(const char *title)
{
    return UserValues[value];
}