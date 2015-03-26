#ifndef CO_MUI_ELEMENT_H
#define CO_MUI_ELEMENT_H

/*! file
 * \brief user interface proxy class
 *
 * \author Andreas Grimm
 * \date
 */

#include <tui/coAbstractTabletUI.h>
#include <OpenVRUI/coUIContainer.h>
#include <cover/mui/support/coMUIListener.h>

class coMUIConfigManager;

namespace opencover
{
class coTUIElement;
}

/*
 *Base class for MUI elements
 */
class COVEREXPORT coMUIElement
{
public:
    // constructor:
    // l: shown Name
    coMUIElement();
    coMUIElement(const std::string &n);

    struct device
    {
        std::string UI;
        std::string Device;
        std::string Identifier;
        bool Visible;
        std::string Label;
    };

    // destructor:
    ~coMUIElement();

    // methods:

    virtual void setEventListener(coMUIListener *l);
    virtual coMUIListener *getMUIListener();

    // must be overwritten, if inherited:
    virtual void setPos(int posx, int posy)=0;
    virtual opencover::coTUIElement* getTUI()=0;
    virtual std::string getUniqueIdentifier()=0;

private:
    coMUIConfigManager *ConfigManager;
protected:
    std::string label_str;                      //< label of the elements
    std::string UniqueIdentifier;
    coMUIListener *listener;

};


#endif
