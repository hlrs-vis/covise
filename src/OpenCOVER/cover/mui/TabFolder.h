// class, which creates a menuentry and a new submenu in VR
// creates a new tab in TUI


#ifndef TABFOLDER_H
#define TABFOLDER_H

#include <cover/coVRPlugin.h>
#include "Container.h"
#include <vector>
#include <boost/smart_ptr.hpp>

// forwarddeclaration

namespace vrui
{
class coSubMenuItem;
class coRowMenu;
class coMenuItem;
}

namespace mui
{
class ConfigManager;
// class:
class COVEREXPORT TabFolder: public Container
{

public:
    // methods:
    static mui::TabFolder* create(std::string uniqueIdentifier, mui::Container* parent = NULL);

    // destructor:
    ~TabFolder();

private:
    // methods:
    // constructor:
    TabFolder(const std::string identifier, Container* parent = NULL);


    // variables:
    boost::shared_ptr<opencover::coTUITab> TUITab;
};
} // end namespace
#endif
