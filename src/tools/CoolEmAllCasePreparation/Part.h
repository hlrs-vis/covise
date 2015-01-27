/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef PART_H
#define PART_H

#include <util/coExport.h>
#include <osg/Referenced>
#include <string>
#include <map>
#include <cstring>

class Part
{

    std::string path;
    std::string p_id;
    std::string m_filename;

public:
    void setName(const char *name);

    void setId(const char *id);

    std::string getName();

    std::string getId();

    std::string makeFileName();
};

#endif
