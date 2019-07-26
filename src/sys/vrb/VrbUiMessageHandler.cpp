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

#include <vrbclient/SharedStateSerializer.h>
#include <vrbclient/SessionID.h>



#include <qsocketnotifier.h>
#define IOMANIPH
 // don't include iomanip.h becaus it interferes with qt

using namespace covise;

void VrbUiMessageHandler::updateApplicationWindow(const char * cl, int sender, const char * var, const covise::DataHandle& value)
{
    char * charVal;
	TokenBuffer tb(value);
    if (strcmp(cl, "SharedState") == 0)
    {
		appwin->registry->updateEntry(cl, sender, var, vrb::tokenBufferToString(std::move(tb)).c_str());
    }
	else if (strcmp(cl, "SharedMap") == 0)
	{
		std::string t("SharedMap");
		appwin->registry->updateEntry(cl, sender, var, t.c_str());
	}
    else
    {
		tb >> charVal;
		appwin->registry->updateEntry(cl, sender, var, charVal);
    }
}

void VrbUiMessageHandler::removeEntryFromApplicationWindow(const char * cl, int sender, const char * var)
{
    appwin->registry->removeEntry(cl, sender, var);
}

void VrbUiMessageHandler::removeEntriesFromApplicationWindow(int sender)
{
    appwin->registry->removeEntries(sender);
}

bool VrbUiMessageHandler::setClientNotifier(covise::Connection * conn, bool state)
{
    if (VrbUiClient *cl = static_cast<VrbUiClient *>(vrbClients->get(conn)))
    {
        cl->getSN()->setEnabled(state);
        return true;
    }
    return false;
}
