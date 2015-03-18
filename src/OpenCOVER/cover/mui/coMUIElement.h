#ifndef CO_MUI_ELEMENT_H
#define CO_MUI_ELEMENT_H

/*! file
 * \brief user interface proxy class
 *
 * \author Andreas Grimm
 * \date
 */

#include <tui/coAbstractTabletUI.h>
#include <QObject>
#include <OpenVRUI/coUIContainer.h>

namespace coMUI
{
class coMUIElement;
}

class coMUIConfigManager;

/*
 *Base class for MUI elements
 */
class COVEREXPORT coMUIElement: public QObject
{

    Q_OBJECT

public:
    // constructor:
    // l: shown Name
    coMUIElement();
    coMUIElement(const std::string &n);

    struct device{
        std::string UI;
        std::string Device;
        std::string Identifier;
        bool Visible;
        std::string Label;
    };

    // destructor:
    ~coMUIElement();

    // methods:

    // must be overwritten, if inherited:
    virtual void setPos(int posx, int posy)=0;
    std::string getUniqueIdentifier();

private:
    coMUIConfigManager *ConfigManager;
protected:
    std::string label_str;                      //< label of the elements
    std::string UniqueIdentifier;

};


#endif
