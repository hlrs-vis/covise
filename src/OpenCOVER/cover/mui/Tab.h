// class, wich creats a menuentry and a new menu as VRUI-Element
// creates a new Tab as TUI-Element

#ifndef MUITAB_H
#define MUITAB_H

#include <cover/coTabletUI.h>
#include "support/ConfigManager.h"
#include "Container.h"
#include <vector>
#include <boost/smart_ptr.hpp>


// forwarddeclaration
namespace opencover
{
class coTUITab;
}

namespace vrui
{
class coSubMenuItem;
class coRowMenu;
}

namespace mui
{
/**
 * @brief The Tab class
 * creates a TUI-Tab-Element and a new submenu in VRUI
 */
class COVEREXPORT Tab: public mui::Container
{

public:
    // methods:
    static mui::Tab* create(std::string uniqueIdentifier, mui::Container* parent = NULL);

    // destructor:
    ~Tab();

private:
    // methods:
    // constructor:
    Tab(const std::string UniqueIdentifier, Container* parent=NULL);
};
} // end namespace
#endif

