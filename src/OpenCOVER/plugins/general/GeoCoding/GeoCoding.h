/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GEOCODING_H
#define GEOCODING_H

#include <string_view>
#include <cover/coVRPlugin.h>
#include <cover/ui/Manager.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Group.h>
#include <cover/ui/Owner.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Label.h>
#include <cover/ui/Button.h>

class GeoCoding : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    GeoCoding();
    bool init();
    ~GeoCoding();

    static GeoCoding *instance();

    void jumpToAddress(std::string_view searchQuery);
    void geocode();

private:
    static GeoCoding *s_instance;

    opencover::ui::Menu *m_geoDataMenu;
    opencover::ui::Group *m_geoCodingGroup;
    opencover::ui::EditField *m_searchQueryField;
    opencover::ui::Label *m_currentLocationLabel;
    opencover::ui::Button *m_actionGeocode;
};
#endif
