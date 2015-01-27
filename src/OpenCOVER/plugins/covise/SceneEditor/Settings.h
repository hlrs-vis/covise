/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SETTINGS_H
#define SETTINGS_H

class Settings
{
public:
    static Settings *instance();
    virtual ~Settings();

    bool isOperatingRangeVisible();
    void setOperatingRangeVisible(bool v);
    bool toggleOperatingRangeVisible();

    bool isGridVisible();
    void setGridVisible(bool v);
    bool toggleGridVisible();

private:
    Settings();

    bool _operatingRangeVisible;
    bool _gridVisible;
};

#endif
