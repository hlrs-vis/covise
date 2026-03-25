#pragma once
#include <lib/core/ComponentBase.h>
#include <cover/ui/Owner.h>

class CoverOwner : public opencover::ui::Owner, public core::ComponentBase
{
public:
    CoverOwner(const std::string &name, opencover::ui::Manager *manager)
        : opencover::ui::Owner(name, manager)
        , core::ComponentBase(nullptr, name)
    {
    }
};
