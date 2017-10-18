/* this Plugin does nothing. It is just an example
 *
 *
 *
*/

#ifndef USERINTERFACEEXAMPLEPLUGIN_H
#define USERINTERFACEEXAMPLEPLUGIN_H

namespace opencover
{
namespace ui
{
class Button;
class Menu;
class Group;
class Slider;
class Label;
}
}

#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>

using namespace opencover;

class UserInterface: public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    UserInterface();
    ~UserInterface();
    bool init();

private:
    ui::Menu *Tab1 = nullptr;

    ui::Button *Button1 = nullptr, *Button2 = nullptr;
    ui::Slider *ValueRegulator1 = nullptr;
    ui::Group *Frame = nullptr;
    ui::Label *Label = nullptr;

};
#endif
