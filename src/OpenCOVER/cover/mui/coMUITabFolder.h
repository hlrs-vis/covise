// class, which creates a menuentry and a new submenu in VR
// creates a new tab in TUI


#ifndef COMUITABFOLDER_H
#define COMUITABFOLDER_H

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

namespace vrui
{
class coSubMenuItem;
class coRowMenu;
class coMenuItem;
}

// class:
class COVEREXPORT coMUITabFolder: public coMUIContainer, public opencover::coTUIListener, public vrui::coMenuListener
{

public:
    // methods:
    // constructor/destructor:
    coMUITabFolder(const std::string identifier, coMUIContainer* parent, std::string label);
    coMUITabFolder(const std::string identifier, coMUIContainer* parent);
    coMUITabFolder(const std::string identifier, std::string label);
    coMUITabFolder(const std::string identifier);

    ~coMUITabFolder();

    void setPos(int posx, int posy);                 // sets pos only for TUI-Element
    int getTUIID();
    opencover::coTUIElement* getTUI();
    vrui::coMenu* getVRUI();
    void setLabel(std::string label);                // sets the label of all UI-Elements
    void setLabel(std::string label, std::string UI);// sets the label of all named UI-Elements
    void setVisible(bool visible, std::string UI);   // sets the Visible-value of all named UI-Elements
    void setVisible(bool visible);                   // sets the Visible-value of all UI-Elements
    coMUIContainer *getParent();
    std::string getUniqueIdentifier();

private:
    // methods:
    void ParentConstructor(const std::string identifier, coMUIContainer* parent); // constructor with parents
    void constructor(const std::string identifier);        // constructor without parents

    // variables:
    coMUIContainer* Parent;
    boost::shared_ptr<opencover::coTUITab> TUITab;
    boost::shared_ptr<opencover::coTUITabFolder> TUIElement;
    boost::shared_ptr<vrui::coSubMenuItem> SubmenuItem;
    boost::shared_ptr<vrui::coRowMenu> Submenu;
    coMUIConfigManager *ConfigManager;
    std::vector<device> Devices;
    std::string Label;
    std::string Identifier;
};

#endif
