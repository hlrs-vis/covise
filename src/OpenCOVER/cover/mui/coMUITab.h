// class, wich creats a menuentry and a new menu as VRUI-Element
// creates a new Tab as TUI-Element

#ifndef COMUITAB_H
#define COMUITAB_H

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenu.h>
#include "support/coMUIConfigManager.h"
#include "coMUIContainer.h"
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

// class:
class COVEREXPORT coMUITab: public coMUIContainer, public opencover::coTUIListener, public vrui::coMenuListener
{

public:
    // methods:
    // constructor/destructor:
    // std::string as input
    coMUITab(const std::string UniqueIdentifier, coMUIContainer* parent,std::string label);
    coMUITab(const std::string UniqueIdentifier, coMUIContainer* parent);
    coMUITab(const std::string UniqueIdentifier, std::string label);
    coMUITab(const std::string UniqueIdentifier);

    ~coMUITab();

    void setPos(int posx, int posy);                 // sets position of TUI-Element
    int getTUIID();
    opencover::coTUIElement* getTUI();

    vrui::coMenu* getVRUI();
    void setLabel(std::string label);                // sets the label of all UI-Elements
    void setLabel(std::string label, std::string UI);// sets the label of the named UI-Elements
    void setVisible(bool visible, std::string UI);   // sets the visible-value of named UI-Elements
    void setVisible(bool visible);                   // sets the visible-value of all UI-Elements
    std::string getUniqueIdentifier();
    coMUIContainer *getParent();

private:
    // methods:
    void ParentConstructor(const std::string identifier, coMUIContainer* parent);   // underlying constructor
    void constructor(const std::string identifier);        // underlying constructor

    // variables:
    coMUIContainer* Parent;
    boost::shared_ptr<opencover::coTUITab> TUIElement;
    boost::shared_ptr<vrui::coSubMenuItem> SubmenuItem;
    boost::shared_ptr<vrui::coRowMenu> Submenu;
    coMUIConfigManager *ConfigManager;
    std::vector<device> Devices;
    std::string Identifier;
    std::string Label;
};

#endif

