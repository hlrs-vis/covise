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

using namespace opencover;




class TestTfm : public coVRPlugin, public ui::Owner
{
public:
    TestTfm();
private:
    ui::Menu * m_menu;
    ui::Slider *m_rotator;
};


#endif
