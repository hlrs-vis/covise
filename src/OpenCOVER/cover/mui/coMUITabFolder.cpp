// class, which creates a menuentry and a new submenu in VR
// creates a new tab in TUI

#include <cover/coVRTui.h>
#include <OpenVRUI/coRowMenu.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenuItem.h>
#include "support/coMUISupport.h"
#include "support/coMUIConfigManager.h"
#include "coMUITabFolder.h"

#include <iostream>

using namespace opencover;
using namespace vrui;

// constructors (with parent):
coMUITabFolder::coMUITabFolder(const std::string UniqueIdentifier, coMUIContainer* parent,  const std::string label){
    ConfigManager= NULL;
    Label = label;
    ParentConstructor(UniqueIdentifier, parent);
}
coMUITabFolder::coMUITabFolder(const std::string UniqueIdentifier, coMUIContainer* parent){
    ConfigManager= NULL;
    Label = UniqueIdentifier;
    ParentConstructor(UniqueIdentifier, parent);
}
// constructors (wihtout parents -> new entry in main-menu);
coMUITabFolder::coMUITabFolder(const std::string UniqueIdentifier, std::string label){

    std::cout << "coMUITabFolder: 0.0" << std::endl;
    ConfigManager= NULL;
    Label = label;

    std::cout << "coMUITabFolder: 0.1" << std::endl;
    constructor(UniqueIdentifier);

    std::cout << "coMUITabFolder: 0.2" << std::endl;
}
coMUITabFolder::coMUITabFolder(const std::string UniqueIdentifier){
    ConfigManager= NULL;
    Label = UniqueIdentifier;
    constructor(UniqueIdentifier);
}

// destructor:
coMUITabFolder::~coMUITabFolder(){
ConfigManager->deletePos(Identifier);
ConfigManager->removeElement(Identifier);
}

// underlying constructors:
void coMUITabFolder::ParentConstructor(const std::string UniqueIdentifier,  coMUIContainer* parent){
    ConfigManager= coMUIConfigManager::getInstance();

    Parent=parent;
    Identifier=UniqueIdentifier;

    ConfigManager->addElement(UniqueIdentifier, this);

    // create default-values or tanken from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device=ConfigManager->keywordCAVE();
    Devices[0].UI=ConfigManager->keywordVRUI();
    Devices[0].Identifier = UniqueIdentifier;
    Devices[0].Visible = true;

    Devices[0].Label= ConfigManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);
    Parent=ConfigManager->getCorrectParent(Parent, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);
    // create VRUI-Element:
    Submenu.reset(new vrui::coRowMenu(Devices[0].Label.c_str()));
    SubmenuItem.reset(new vrui::coSubMenuItem(Devices[0].Label));

    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[1].Device=ConfigManager->keywordTablet();
    Devices[1].UI=ConfigManager->keywordTUI();
    Devices[1].Identifier = UniqueIdentifier;
    Devices[1].Visible = true;

    Devices[1].Label = ConfigManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].Identifier);
    Parent=ConfigManager->getCorrectParent(Parent, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);
    // create TUI-Element:
    TUIElement.reset(new coTUITabFolder(Devices[1].Label, Parent->getTUIID()));


    // find and set correct parameter (get them from configuration file, if possible):
    for (int i=0; i<Devices.size(); ++i){
        Devices[i].Visible = ConfigManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
        Devices[i].Label = ConfigManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);

        // create the UI-Elements
        if  (Devices[i].UI==ConfigManager->keywordTUI()){                       // create TUI-Elements
            int posx=ConfigManager->getCorrectPosX(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
            int posy=ConfigManager->getCorrectPosY(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, Parent->getUniqueIdentifier());
            TUIElement->setPos(posx,posy);
            ConfigManager->addPos(Devices[i].Identifier, posx, posy, Parent->getUniqueIdentifier(), true);
            TUIElement->setHidden(!Devices[i].Visible);
        }
        else if (Devices[i].UI==ConfigManager->keywordVRUI()){                  // create VRUI-Elements
            SubmenuItem->setMenu(Submenu.get());

            if (Devices[i].Visible){                                            // visible
                ConfigManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->add(SubmenuItem.get());
            }
            else{                                                               // invisible
                ConfigManager->getCorrectParent(parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier)->getVRUI()->remove(SubmenuItem.get());
            }
        } else {
            std::cerr << "coMUITabFolder::ParentConstructor(): " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}

void coMUITabFolder::constructor(const std::string identifier){

    ConfigManager= coMUIConfigManager::getInstance();
    ConfigManager->addElement(identifier, this);
    Identifier=identifier;

    // create defaultvalue or take from constructor:
    // VRUI-CAVE-Element:
    Devices.push_back(device());
    Devices[0].Device=ConfigManager->keywordCAVE();
    Devices[0].UI=ConfigManager->keywordVRUI();
    Devices[0].Identifier = identifier;

    Devices[0].Label= ConfigManager->getCorrectLabel(Label, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);
    Parent = ConfigManager->getCorrectParent(NULL, Devices[0].UI, Devices[0].Device, Devices[0].Identifier);
    // create VRUI-Element:
    Submenu.reset(new vrui::coRowMenu(Devices[0].Label.c_str()));
    SubmenuItem.reset(new vrui::coSubMenuItem(Devices[0].Label));

    // TUI-Tablet-Element:
    Devices.push_back(device());
    Devices[1].Device=ConfigManager->keywordTablet();
    Devices[1].UI=ConfigManager->keywordTUI();
    Devices[1].Identifier = identifier;
    Devices[1].Label = ConfigManager->getCorrectLabel(Label, Devices[1].UI, Devices[1].Device, Devices[1].Identifier);
    Parent = ConfigManager->getCorrectParent(NULL, Devices[1].UI, Devices[1].Device, Devices[1].Identifier);
    // create TUI-Element:

    if (Parent){                                         // parent was declated in configuration file
        TUITab.reset(new coTUITab(Devices[1].Label, Parent->getTUIID()));
        TUIElement.reset(new coTUITabFolder(Devices[1].Label, TUITab->getID()));
    } else if (!Parent){                                  // no parent declared->mainmenu
        TUITab.reset(new coTUITab(Devices[1].Label, coVRTui::instance()->mainFolder->getID()));
        TUIElement.reset(new coTUITabFolder(Devices[1].Label, TUITab->getID()));
    } else{
        std::cerr << "coMUITabFolder::constructor(): Parent not found" << std::endl;
    }

    // find and set correct parameter (get them from configuration file, if possible):
    for (int i=0; i<Devices.size(); ++i){
        Devices[i].Visible = ConfigManager->getCorrectVisible(Devices[i].Visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
        Devices[i].Label = ConfigManager->getCorrectLabel(Label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);

        // create UI-Elements:
        if  (Devices[i].UI==ConfigManager->keywordTUI()){  // create TUI-Element
            int posx=ConfigManager->getCorrectPosX(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, ConfigManager->keywordMainWindow());
            int posy=ConfigManager->getCorrectPosY(Devices[i].UI, Devices[i].Device, Devices[i].Identifier, ConfigManager->keywordMainWindow());
            TUITab->setPos(posx,posy);
            ConfigManager->addPos(Devices[i].Identifier, posx, posy, ConfigManager->keywordMainWindow(), true);
            TUITab->setHidden(!Devices[i].Visible);
        }
        else if (Devices[i].UI==ConfigManager->keywordVRUI()){  // create VRUI-Elemente erstellen
            SubmenuItem->setMenu(Submenu.get());

            if (Devices[i].Visible){                                            // visible
                Parent=ConfigManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
                if (!Parent){
                    cover->getMenu()->add(SubmenuItem.get());
                } else if (Parent){
                    Parent->getVRUI()->add(SubmenuItem.get());
                } else {
                    std::cerr << "coMUITabFolder::constructor: wrong Parent" << std::endl;
                }
            }
            else{                                                               // invisible
                Parent=ConfigManager->getCorrectParent(Parent, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
                if (!Parent){
                    cover->getMenu()->remove(SubmenuItem.get());
                } else if (Parent){
                    Parent->getVRUI()->remove(SubmenuItem.get());
                } else {
                    std::cerr << "coMUITabFolder::constructor: wrong Parent" << std::endl;
                }
            }
        } else {
            std::cerr << "coMUITabFolder::constructor: " << Devices[i].UI << " not found in Constructor." << std::endl;
        }
    }
}


//  returns the ID of the TUIElements
int coMUITabFolder::getTUI()
{
    return TUIElement->getID();
}

// returns the VRUI-Parent
coMenu* coMUITabFolder::getVRUI()
{
    return Submenu.get();
}

// sets the label for all UI-elements
void coMUITabFolder::setLabel(std::string label){
    std::string UI;
    for (int i=0; i<Devices.size(); ++i){
        UI.append(Devices[i].UI + " ");
    }
    setLabel(label, UI);
}

// sets the Label for all named Elements
void coMUITabFolder::setLabel(std::string label, std::string UI){
    for (int i=0; i<Devices.size(); ++i){
        if (UI.find(Devices[i].UI) != std::string::npos){                       // Element to be changed
            Devices[i].Label=ConfigManager->getCorrectLabel(label, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            if (Devices[i].UI == ConfigManager->keywordTUI()){                  // TUI-Element
                TUITab->setLabel(Devices[i].Label);
            } else if (Devices[i].UI == ConfigManager->keywordVRUI()){          // VRUI-Element
                SubmenuItem->setLabel(Devices[i].Label);
                Submenu->updateTitle(Devices[i].Label.c_str());
            } else{
                std::cerr << "coMUIMainElement::setLabel(): Elementtype " << Devices[i].UI << " not found in setLabel(std::string, std::string)." << std::endl;
            }
        }
    }
}

// sets the visibility for all elements
void coMUITabFolder::setVisible(bool visible){
    std::string UI;
    for (int i=0; i<Devices.size(); ++i){
        UI.append(Devices[i].UI + " ");
    }
    setVisible(visible, UI);
}

// sets the visibility for the named elements
void coMUITabFolder::setVisible(bool visible, std::string UI){
    for (int i=0; i<Devices.size(); ++i){
        if (UI.find(Devices[i].UI)!=std::string::npos){                             // Element to be changed
            if (Devices[i].Visible != visible){
                Devices[i].Visible = ConfigManager->getCorrectVisible(visible, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
                if (Devices[i].UI == ConfigManager->keywordTUI()){                  // TUI-Element
                    TUITab->setHidden(!Devices[i].Visible);
                } else if (Devices[i].UI == ConfigManager->keywordVRUI()){          // VRUI-Element
                    if (Devices[i].Visible){
                        cover->getMenu()->add(SubmenuItem.get());
                    } else{
                        cover->getMenu()->remove(SubmenuItem.get());
                    }
                } else{
                    std::cerr << "coMUITabFolder::setVisible(): Elementtyp " << Devices[i].UI << " wurde in setVisible(string, bool, bool) nicht gefunden." << std::endl;
                }
            }
        }
    }
}

// sets Position
void coMUITabFolder::setPos(int posx, int posy){
    for (int i=0; i<Devices.size(); ++i){
        if (Devices[i].UI == ConfigManager->keywordTUI()){                      // TUI-Element
            int posX=ConfigManager->getCorrectPosX(posx, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            int posY=ConfigManager->getCorrectPosY(posy, Devices[i].UI, Devices[i].Device, Devices[i].Identifier);
            ConfigManager->preparePos(posx,posy, Parent->getUniqueIdentifier());
            TUITab->setPos(posX,posY);
        }
    }
}

std::string coMUITabFolder::getUniqueIdentifier(){
    return Devices[0].Identifier;
}
