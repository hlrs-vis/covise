/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "VrbUiMessageHandler.h"
#include "VRBServer.h"
#include "VrbUiClientList.h"

#include "gui/VRBapplication.h"
#include "gui/coRegister.h"

#include <net/message.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <net/dataHandle.h>

#include <net/tokenbuffer_serializer.h>
#include <vrb/SessionID.h>

#include <cassert>

#define IOMANIPH
 // don't include iomanip.h becaus it interferes with qt

using namespace covise;


void VrbUiMessageHandler::updateApplicationWindow(const std::string& cl, int sender, const std::string& var, const covise::DataHandle& value)
{
    char * charVal;
	TokenBuffer tb(value);
    if (cl == "SharedState")
    {
		appwin->registry->updateEntry(cl.c_str(), sender, var.c_str(), covise::tokenBufferToString(std::move(tb)).c_str());
    }
	else if (cl == "SharedMap")
	{
		std::string t("SharedMap");
		appwin->registry->updateEntry(cl.c_str(), sender, var.c_str(), t.c_str());
	}
    else
    {
		tb >> charVal;
		appwin->registry->updateEntry(cl.c_str(), sender, var.c_str(), charVal);
    }
}

void VrbUiMessageHandler::removeEntryFromApplicationWindow(const std::string& cl, int sender, const std::string& var)
{
    appwin->registry->removeEntry(cl.c_str(), sender, var.c_str());
}

void VrbUiMessageHandler::removeEntriesFromApplicationWindow(int sender)
{
    appwin->registry->removeEntries(sender);
}

bool VrbUiMessageHandler::setClientNotifier(const covise::Connection * conn, bool state)
{
    if (VrbUiClient *cl = static_cast<VrbUiClient *>(vrbClients->get(conn)))
    {
        cl->getSN()->setEnabled(state);
        return true;
    }
    return false;
}

vrb::VRBSClient *VrbUiMessageHandler::createNewClient(vrb::ConnectionDetails::ptr &&cd, covise::TokenBuffer &tb){
    auto uicd = dynamic_cast<UiConnectionDetails*>(cd.get());
    assert(uicd);
    return new VrbUiClient(cd->tcpConn, cd->udpConn, uicd->notifier.release(), tb);
}
