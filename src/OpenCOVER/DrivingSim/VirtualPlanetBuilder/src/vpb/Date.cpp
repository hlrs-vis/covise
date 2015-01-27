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

#include <vpb/Date>
#include <vpb/FileUtils>

#include <time.h>

// for struct stat
#include <sys/types.h>
#include <sys/stat.h>

using namespace vpb;

bool Date::setWithDateOfLastModification(const std::string &filename)
{
    struct stat s;
    int status = stat(filename.c_str(), &s);
    if (status == 0)
    {
        const struct tm *tm_date = localtime(&(s.st_mtime));

        year = 1900 + tm_date->tm_year;
        month = tm_date->tm_mon;
        day = tm_date->tm_mday;
        hour = tm_date->tm_hour;
        minute = tm_date->tm_min;
        second = tm_date->tm_sec;

        return true;
    }
    else
    {
        return false;
    }
}

bool Date::setWithCurrentDate()
{
    time_t t = time(NULL);
    struct tm *tm_date = localtime(&t);
    if (tm_date)
    {
        year = 1900 + tm_date->tm_year;
        month = tm_date->tm_mon;
        day = tm_date->tm_mday;
        hour = tm_date->tm_hour;
        minute = tm_date->tm_min;
        second = tm_date->tm_sec;
        return true;
    }
    else
    {
        return false;
    }
}

bool Date::setWithDateString(const std::string &dateString)
{
    int result = sscanf(dateString.c_str(), "%u/%u/%u %u:%u:%u", &year, &month, &day, &hour, &minute, &second);
    return result == 6;
}

std::string Date::getDateString() const
{
    char datestring[256];
    sprintf(datestring, "%u/%u/%u %u:%u:%u", year, month, day, hour, minute, second);
    return std::string(datestring);
}
