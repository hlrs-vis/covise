/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef testTFM_H
#define testTFM_H

#include <map>


#include <cover/coVRPlugin.h>

#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Slider.h>

#include "../Bicycle/AVRserialcom.h"

using namespace opencover;




class Drehgeber : public coVRPlugin, public ui::Owner,  public OpenThreads::Thread
{
public:
    Drehgeber();
    ~Drehgeber();

    bool init();
    void run();
    bool update();
private:
    ui::Menu * m_menu;
    ui::Slider *m_rotator;
    ui::TextField* serialDeviceUI;
    ui::EditField* baudrateUI;
    bool running = true;
    int counter;
    float angle;
    std::string SerialDev;

    std::unique_ptr<ConfigString> SerialDevice;
    std::unique_ptr<ConfigInt> baudrate;
};


#endif
