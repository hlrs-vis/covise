// class, wich creats a menuentry and a new menu as VRUI-Element
// creates a new Tab as TUI-Element

#ifndef MUITAB_H
#define MUITAB_H

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenu.h>
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
// class:
class COVEREXPORT Tab: public Container, public opencover::coTUIListener, public vrui::coMenuListener
{

public:
    // methods:
    // constructor/destructor:
    // std::string as input
    Tab(const std::string UniqueIdentifier, Container* parent,std::string label);
    Tab(const std::string UniqueIdentifier, Container* parent);
    Tab(const std::string UniqueIdentifier, std::string label);
    Tab(const std::string UniqueIdentifier);

    ~Tab();

    void setPos(int posx, int posy);                 // sets position of TUI-Element
    int getTUIID();
    opencover::coTUIElement* getTUI();

    vrui::coMenu* getVRUI();
    void setLabel(std::string label);                // sets the label of all UI-Elements
    void setLabel(std::string label, std::string UI);// sets the label of the named UI-Elements
    void setVisible(bool visible, std::string UI);   // sets the visible-value of named UI-Elements
    void setVisible(bool visible);                   // sets the visible-value of all UI-Elements
    std::string getUniqueIdentifier();
    Container *getParent();

private:
    // methods:
    void ParentConstructor(const std::string identifier, Container* parent);   // underlying constructor
    void constructor(const std::string identifier);        // underlying constructor

    // variables:
    Container* Parent;
    boost::shared_ptr<opencover::coTUITab> TUIElement;
    boost::shared_ptr<vrui::coSubMenuItem> SubmenuItem;
    boost::shared_ptr<vrui::coRowMenu> Submenu;
    ConfigManager *configManager;
    std::vector<device> Devices;
    std::string Identifier;
    std::string Label;
};
} // end namespace
#endif

