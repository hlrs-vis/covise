// class, which creates a new menu as VRUI
// creates a Frame as TUI


#ifndef COMUIFRAME_H
#define COMUIFRAME_H

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenuItem.h>
#include "coMUIContainer.h"
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

class coMUIConfigManager;

class COVEREXPORT coMUIFrame: public coMUIContainer, public opencover::coTUIListener, public vrui::coMenuListener
{

public:
    // constructor/destructor
    coMUIFrame(const std::string UniqueIdentifier, coMUIContainer* parent, std::string label);
    coMUIFrame(const std::string UniqueIdentifier, coMUIContainer* parent);
    ~coMUIFrame();

    // methods:
    int getTUIID();
    opencover::coTUIElement* getTUI();
    vrui::coMenu* getVRUI();
    void setLabel(std::string label);            // set label for all UI-Elements
    void setLabel(std::string label, std::string UI);  // set label for named UI-Elements
    void setVisible(bool visible);               // set visible for all UI-Elements
    void setVisible(bool visible, std::string UI); // set visible for named UI-Elements
    void setPos(int posx, int posy);           // positioning TUI-Element
    coMUIContainer* getParent();                            // returns the parent
    std::string getUniqueIdentifier();

    // variables:

private:
    // methods:
    void constructor(const std::string UniqueIdentifier, coMUIContainer* parent, std::string label);
    // variables:
    std::string Label;
    std::string Identifier;
    std::vector<device>Devices;
    boost::shared_ptr<vrui::coRowMenu> Submenu;
    boost::shared_ptr<vrui::coSubMenuItem> SubmenuItem;
    boost::shared_ptr<opencover::coTUIFrame> TUIElement;
    coMUIConfigManager *ConfigManager;
    coMUIContainer* Parent;
};

#endif
