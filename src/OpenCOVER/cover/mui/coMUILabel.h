// class, which creates a label in VRUI and TUI

#ifndef COMUILABEL_H
#define COMUILABEL_H

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include "coMUIWidget.h"
#include <boost/smart_ptr.hpp>

namespace vrui
{
class coLabelMenuItem;
}
namespace opencover
{
class coTUILabel;
}

class coMUIConfigManager;
class coMUIContainer;

class COVEREXPORT coMUILabel:public coMUIWidget
{

public:
    // constructor/destructor:
    coMUILabel(std::string UniqueIdentifier, coMUIContainer* parent, std::string label);
    coMUILabel(std::string UniqueIdentifier, coMUIContainer* parent);
    ~coMUILabel();

    // methods:
    std::string getLabel();
    opencover::coTUIElement* getTUI();
    void setPos(int posx, int posy);
    void setLabel(std::string label);
    void setLabel(std::string label, std::string UI);
    void setVisible(bool visible);
    void setVisible(bool visible, std::string UI);
    coMUIContainer* getParent();
    std::string getUniqueIdentifier();


private:
    std::vector<device> Devices;
    void constructor(std::string UniqueIdentifier, coMUIContainer* parent, std::string label);
    std::string Label;
    std::string Identifier;
    boost::shared_ptr<opencover::coTUILabel> TUIElement;
    boost::shared_ptr<vrui::coLabelMenuItem> VRUIElement;
    coMUIConfigManager *ConfigManager;
    coMUIContainer* Parent;

    void changeLabel(std::string label);        // changes the label

    void changedLabel();                        // emmitted, if the label changed
};


#endif
