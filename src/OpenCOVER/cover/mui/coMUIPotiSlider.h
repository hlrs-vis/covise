// class, which creates a Poti as VRUI-Element
// creates a Slider as TUI-Element

#ifndef COMUIPOTISLIDER_H
#define COMUIPOTISLIDER_H

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <OpenVRUI/coMenuItem.h>
#include "coMUIWidget.h"
#include <boost/smart_ptr.hpp>

namespace vrui
{
class coPotiMenuItem;
class coRowMenu;
class coUIElement;
class coMenu;
}

// forwarddeclaration:
class coMUIConfigManager;
class coMUIContainer;

class COVEREXPORT coMUIPotiSlider:public coMUIWidget, public opencover::coTUIListener, public vrui::coMenuListener{
    Q_OBJECT

public:
    // constructor/destructor:
    coMUIPotiSlider(const std::string UniqueIdentifier, coMUIContainer* parent, float min, float max, float defaultValue, const std::string label);
    coMUIPotiSlider(const std::string UniqueIdentifier, coMUIContainer* parent, float min, float max, float defaultValue);
    ~coMUIPotiSlider();

    // methods:

    float getValue();
    void setPos(int posx, int posy);
    opencover::coTUIElement* getTUI();
    void setVisible(bool visible);                   // set visible-value for all UI-Elements
    void setVisible(bool visible, std::string UI);   // set visible-value for named UI-Elements
    void setLabel(std::string label);                // set label for all UI-Elements
    void setLabel(std::string label, std::string UI);// set label for named UI-Elements
    coMUIContainer* getParent();
    std::string getUniqueIdentifier();

    // variables:

private:
    // variables:
    float value;
    float minVal;
    float maxVal;

    std::vector<device> Devices;
    std::string Label;
    std::string Identifier;
    coMUIContainer* Parent;
    coMUIConfigManager *ConfigManager;
    boost::shared_ptr<opencover::coTUIFloatSlider> TUIElement;
    boost::shared_ptr<vrui::coPotiMenuItem> VRUIElement;

    // methods:
    void constructor(const std::string UniqueIdentifier, coMUIContainer* parent, float min, float max, float defaultValue, const std::string label);
    void createTUIElement(std::string Label, coMUIContainer* Parent);
    void createVRUIElement (std::string Label);

    void tabletEvent(opencover::coTUIElement *tUIItem);
    void menuEvent(vrui::coMenuItem *menuItem);

public slots:
    void setValue(float newVal);

signals:
    void valueChanged();

};

#endif
