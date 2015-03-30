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
class ConfigManager;

class COVEREXPORT Frame: public Container, public opencover::coTUIListener, public vrui::coMenuListener
{

public:
    // constructor/destructor
    Frame(const std::string UniqueIdentifier, Container* parent, std::string label);
    Frame(const std::string UniqueIdentifier, Container* parent);
    ~Frame();

    // methods:
    int getTUIID();
    opencover::coTUIElement* getTUI();
    vrui::coMenu* getVRUI();
    void setLabel(std::string label);            // set label for all UI-Elements
    void setLabel(std::string label, std::string UI);  // set label for named UI-Elements
    void setVisible(bool visible);               // set visible for all UI-Elements
    void setVisible(bool visible, std::string UI); // set visible for named UI-Elements
    void setPos(int posx, int posy);           // positioning TUI-Element
    Container* getParent();                            // returns the parent
    std::string getUniqueIdentifier();

    // variables:

private:
    // methods:
    void constructor(const std::string UniqueIdentifier, Container* parent, std::string label);
    // variables:
    std::string Label;
    std::string Identifier;
    std::vector<device>Devices;
    boost::shared_ptr<vrui::coRowMenu> Submenu;
    boost::shared_ptr<vrui::coSubMenuItem> SubmenuItem;
    boost::shared_ptr<opencover::coTUIFrame> TUIElement;
    ConfigManager *configManager;
    Container* Parent;
};
} // end namespace
#endif
