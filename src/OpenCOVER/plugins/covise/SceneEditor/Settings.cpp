/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Settings.h"

Settings *Settings::instance()
{
    static Settings *singleton = 0;
    if (!singleton)
        singleton = new Settings;
    return singleton;
}

Settings::Settings()
{
    _operatingRangeVisible = false;
    _gridVisible = false;
}

Settings::~Settings()
{
}

bool Settings::isOperatingRangeVisible()
{
    return _operatingRangeVisible;
}

void Settings::setOperatingRangeVisible(bool v)
{
    _operatingRangeVisible = v;
}

bool Settings::toggleOperatingRangeVisible()
{
    _operatingRangeVisible = !_operatingRangeVisible;
    return _operatingRangeVisible;
}

bool Settings::isGridVisible()
{
    return _gridVisible;
}

void Settings::setGridVisible(bool v)
{
    _gridVisible = v;
}

bool Settings::toggleGridVisible()
{
    _gridVisible = !_gridVisible;
    return _gridVisible;
}
