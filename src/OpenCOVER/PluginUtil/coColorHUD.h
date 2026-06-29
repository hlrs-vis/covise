#ifndef COVRPLUGINUTIL_COLOR_HUD_H
#define COVRPLUGINUTIL_COLOR_HUD_H

#include <cover/ui/Owner.h>
#include <memory>
#include "ColorBar.h"
struct ColorsModule: public opencover::ui::Owner
{
    ColorsModule(const std::string &name, opencover::ui::Owner *owner)
        : opencover::ui::Owner(name, owner)
    {}

    bool hudVisible() const
    {
        return colorbar && colorbar->hudVisible();
    }
    int useCount = 0;
    std::unique_ptr<opencover::CoviseColorBar> colorbar = nullptr;
    opencover::ui::Menu *menu = nullptr;
};


#endif // COVRPLUGINUTIL_COLOR_HUD_H