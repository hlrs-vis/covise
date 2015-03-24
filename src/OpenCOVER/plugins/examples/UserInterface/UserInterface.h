/* this Plugin does nothing. It is just an example
 *
 *
 *
*/

#ifndef BEISPIEL_H
#define BEISPIEL_H

#include <cover/coVRPlugin.h>
#include <boost/smart_ptr.hpp>

class coMUIToggleButton;
class coMUITabFolder;
class coMUIFrame;
class coMUITab;
class coMUIConfigManager;
class coMUIPotiSlider;
class coMUILabel;

namespace opencover{
class coTUITab;
}


class UserInterface: public opencover::coVRPlugin
{
public:
    UserInterface();
    ~UserInterface();
    bool init();
    coMUIConfigManager *ConfigManager;
private:
    boost::shared_ptr<coMUITab> Tab1;
    boost::shared_ptr<coMUIToggleButton> Button1;
    boost::shared_ptr<coMUIToggleButton> Button2;
    boost::shared_ptr<coMUIPotiSlider> Slider1;
    boost::shared_ptr<coMUIFrame> Frame;
    boost::shared_ptr<coMUILabel> Label;
};

#endif
