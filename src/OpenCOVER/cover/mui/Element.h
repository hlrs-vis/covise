#ifndef MUIELEMENT_H
#define MUIELEMENT_H

/*! file
 * \brief user interface proxy class
 *
 * \author Andreas Grimm
 * \date
 */
#include <cover/coVRPlugin.h>
#include <cover/mui/support/EventListener.h>
#include <cover/mui/support/DefaultValues.h>


namespace opencover
{
class coTUIElement;
}

namespace mui
{
class ConfigManager;

/*
 *Base class for MUI elements
 */
class COVEREXPORT Element
{
public:
    // constructor:
    Element();
    Element(const std::string name);

    struct device
    {
        mui::UITypeEnum UI;
        mui::DeviceTypesEnum Device;
        std::string UniqueIdentifier;
        bool Visible;
        std::string Label;
    };

    // destructor:
    virtual ~Element();

    // methods:

    virtual void setEventListener(EventListener *listener);
    virtual EventListener *getMUIListener();

    // must be overwritten, if inherited:
    virtual void setPos(int posx, int posy)=0;
    virtual opencover::coTUIElement* getTUI()=0;
    virtual std::string getUniqueIdentifier()=0;

private:
    ConfigManager *configManager;
protected:
    std::string label;                      //< label of the elements
    std::string UniqueIdentifier;
    EventListener *Listener;

};
} // end namespace

#endif
