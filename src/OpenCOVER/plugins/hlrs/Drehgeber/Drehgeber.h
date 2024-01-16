/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef testTFM_H
#define testTFM_H

#include <map>
#include <deque>

#include <cover/coVRPlugin.h>

#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Slider.h>

#include <cover/ui/CovconfigLink.h>

#include "../Bicycle/AVRserialcom.h"

using namespace opencover;


class Drehgeber : public coVRPlugin, public ui::Owner,  public OpenThreads::Thread
{
public:
    Drehgeber();
    ~Drehgeber();

    void run();
    bool update();
private:
    std::shared_ptr<config::File>m_config;
    ui::Menu * m_menu;
    ui::Slider *m_rotator;
    std::unique_ptr<ui::EditFieldConfigValue> m_serialDevice;
    std::unique_ptr<ui::EditFieldConfigValue> m_baudrate;
    std::unique_ptr<ui::SliderConfigValue>m_delay;
    bool running = true;
    int counter = 0;
    float angle = 0;
    std::string SerialDev;
    std::deque<std::pair<double, float>> m_values;
};


#endif
