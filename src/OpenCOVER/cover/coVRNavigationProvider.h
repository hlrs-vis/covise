/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVR_NAVIGATION_PROVIDER_H
#define COVR_NAVIGATION_PROVIDER_H

#include <string>

#include <util/common.h>

namespace opencover {

namespace ui {
class Button;
}

class coVRPlugin;

class COVEREXPORT coVRNavigationProvider
{
public:
    coVRNavigationProvider(const std::string name, coVRPlugin* plugin);
    virtual ~coVRNavigationProvider();

    const std::string& getName() const;
    virtual void setEnabled(bool enabled);
    bool isEnabled() const;

    coVRPlugin *plugin;
    ui::Button* navMenuButton = nullptr;
    int ID = -1;

private:
    std::string name;
    bool enabled = false;
};

}

#endif
