// class, which creates a Poti as VRUI-Element
// creates a Slider as TUI-Element

#ifndef MUIPOTISLIDER_H
#define MUIPOTISLIDER_H

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenuItem.h>
#include "Widget.h"
#include <boost/smart_ptr.hpp>

namespace vrui
{
class coPotiMenuItem;
class coRowMenu;
class coUIElement;
class coMenu;
}


namespace mui
{
// forwarddeclaration:
class ConfigManager;
class Container;

class COVEREXPORT PotiSlider:public Widget, public opencover::coTUIListener, public vrui::coMenuListener
{

public:
    // constructor/destructor:
    PotiSlider(const std::string UniqueIdentifier, Container* parent, float min, float max, float defaultValue, const std::string label);
    PotiSlider(const std::string UniqueIdentifier, Container* parent, float min, float max, float defaultValue);
    ~PotiSlider();

    // methods:

    float getValue();
    void setPos(int posx, int posy);
    opencover::coTUIElement* getTUI();
    void setVisible(bool visible);                   // set visible-value for all UI-Elements
    void setVisible(bool visible, std::string UI);   // set visible-value for named UI-Elements
    void setLabel(std::string label);                // set label for all UI-Elements
    void setLabel(std::string label, std::string UI);// set label for named UI-Elements
    Container* getParent();
    std::string getUniqueIdentifier();

    void setValue(float newVal);

    // variables:

private:
    // variables:
    float value;
    float minVal;
    float maxVal;

    std::vector<device> Devices;
    std::string Label;
    std::string Identifier;
    Container* Parent;
    ConfigManager *configManager;
    boost::shared_ptr<opencover::coTUIFloatSlider> TUIElement;
    boost::shared_ptr<vrui::coPotiMenuItem> VRUIElement;

    // methods:
    void constructor(const std::string UniqueIdentifier, Container* parent, float min, float max, float defaultValue, const std::string label);
    void createTUIElement(std::string Label, Container* Parent);
    void createVRUIElement (std::string Label);

    void tabletEvent(opencover::coTUIElement *tUIItem);
    void menuEvent(vrui::coMenuItem *menuItem);



};
} // end namespace
#endif
