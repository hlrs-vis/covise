// class, which creates a label in VRUI and TUI

#ifndef MUILABELELEMENT_H
#define MUILABELELEMENT_H

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include "Widget.h"
#include <boost/smart_ptr.hpp>

namespace vrui
{
class coLabelMenuItem;
}
namespace opencover
{
class coTUILabel;
}

namespace mui
{
class ConfigManager;
class Container;

class COVEREXPORT LabelElement:public Widget
{

public:
    // constructor/destructor:
    LabelElement(std::string UniqueIdentifier, Container* parent, std::string label);
    LabelElement(std::string UniqueIdentifier, Container* parent);
    ~LabelElement();

    // methods:
    std::string getLabel();
    opencover::coTUIElement* getTUI();
    void setPos(int posx, int posy);
    void setLabel(std::string label);
    void setLabel(std::string label, std::string UI);
    void setVisible(bool visible);
    void setVisible(bool visible, std::string UI);
    Container* getParent();
    std::string getUniqueIdentifier();


private:
    std::vector<device> Devices;
    void constructor(std::string UniqueIdentifier, Container* parent, std::string label);
    std::string Label;
    std::string Identifier;
    boost::shared_ptr<opencover::coTUILabel> TUIElement;
    boost::shared_ptr<vrui::coLabelMenuItem> VRUIElement;
    ConfigManager *configManager;
    Container* Parent;

    void changeLabel(std::string label);        // changes the label

    void changedLabel();                        // emmitted, if the label changed
};
} // end namespace

#endif
