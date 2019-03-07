#include "coVrbMenue.h"
#include "coVrbMenue.h"
/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVrbMenue.h"

#include "ui/Action.h"
#include "ui/EditField.h"
#include "ui/Menu.h"
#include "ui/SelectionList.h"

#include "coVRPluginSupport.h"
#include "coVRCommunication.h"

#include <net/tokenbuffer.h>
#include <net/message_types.h>
#include <cassert>

using namespace covise;
namespace opencover
{
VrbMenue::VrbMenue(ui::Owner *owner) 
    :m_owner(owner)
{
    init();
}
void VrbMenue::update(bool state)
{

}
void VrbMenue::init()
{
    menue = new ui::Menu("VrbOptions", m_owner);
    menue->setText("Vrb");

    saveBtn.reset(new ui::Action(menue, "SaveSession"));
    saveBtn->setText("Save session");
    saveBtn->setCallback([this]()
    {
        saveSession();
    });

    loadSL = std::unique_ptr<ui::SelectionList>(new ui::SelectionList(menue, "LoadSession"));
    loadSL->setText("Load session");
    loadSL->setCallback([this](int index)
    {
        if (index == 0)
        {
            unloadAll();
        }
        std::vector<std::string>::iterator it = savedRegistries.begin();
        std::advance(it, index);
        loadSession(*it);
    });

    loadSL->setList(savedRegistries);
    saveBtn->setVisible(true);
    saveBtn->setEnabled(true);
    loadSL->setVisible(true);
    loadSL->setEnabled(true);
}

void VrbMenue::removeVRB_UI()
{
    saveBtn->setVisible(false);
    saveBtn->setEnabled(false);
    loadSL->setVisible(false);
    loadSL->setEnabled(false);
}

void VrbMenue::saveSession()
{
    assert(getPrivateSessionID() != 0);
    TokenBuffer tb;
    if (getPublicSessionID() == 0)
    {
        tb << getPrivateSessionID();
    }
    else
    {
        tb << getPublicSessionID();
    }
    if (vrbc)
    {
        vrbc->sendMessage(tb, COVISE_MESSAGE_VRB_SAVE_SESSION);
    }
}

void VrbMenue::loadSession(std::string &filename)
{
    TokenBuffer tb;
    tb << getID();
    if (getPublicSessionID() == 0)
    {
        tb << getPrivateSessionID();
    }
    else
    {
        tb << getPublicSessionID();
    }
    tb << filename;
    if (vrbc)
    {
        vrbc->sendMessage(tb, COVISE_MESSAGE_VRB_LOAD_SESSION);
    }
}

void VrbMenue::unloadAll()
{
}


}