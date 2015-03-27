#ifndef MUIELEMENT_H
#define MUIELEMENT_H

/*! file
 * \brief user interface proxy class
 *
 * \author Andreas Grimm
 * \date
 */

#include <tui/coAbstractTabletUI.h>
#include <OpenVRUI/coUIContainer.h>
#include <cover/mui/support/Listener.h>


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
    // l: shown Name
    Element();
    Element(const std::string &n);

    struct device
    {
        std::string UI;
        std::string Device;
        std::string Identifier;
        bool Visible;
        std::string Label;
    };

    // destructor:
    ~Element();

    // methods:

    virtual void setEventListener(Listener *l);
    virtual Listener *getMUIListener();

    // must be overwritten, if inherited:
    virtual void setPos(int posx, int posy)=0;
    virtual opencover::coTUIElement* getTUI()=0;
    virtual std::string getUniqueIdentifier()=0;

private:
    ConfigManager *configManager;
protected:
    std::string label_str;                      //< label of the elements
    std::string UniqueIdentifier;
    Listener *listener;

};
} // end namespace

#endif
