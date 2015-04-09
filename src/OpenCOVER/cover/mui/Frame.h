// class, which creates a new menu as VRUI
// creates a Frame as TUI


#ifndef MUIFRAME_H
#define MUIFRAME_H

#include <cover/coTabletUI.h>
#include "Container.h"
#include <boost/smart_ptr.hpp>



// forwarddeclaration
namespace vrui
{
class coRowMenu;
class coSubMenuItem;
}
namespace opencover
{
class coTUIFrame;
}

namespace mui
{
/**
 * @brief The Frame class
 * creates a TUI-Frame-Element and a new submenu in VRUI
 */
class COVEREXPORT Frame: public mui::Container
{

public:
    // methods:
    static mui::Frame* create(std::string uniqueIdentifier, mui::Container* parent);

    // destructor
    ~Frame();

private:
    // methods:
    // constructor:
    Frame(const std::string UniqueIdentifier, Container* parent);
};
} // end namespace
#endif
