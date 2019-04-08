/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#include "VrbUiMessageHandler.h"
#include <net/message.h>
#include <net/message_types.h>
#include <net/tokenbuffer.h>
#include <vrbclient/SharedStateSerializer.h>
#include <vrbclient/SessionID.h>
#include "VRBServer.h"
#include "VrbUiClientList.h"
#include "gui/VRBapplication.h"
#include "gui/coRegister.h"

#include <qsocketnotifier.h>
#define IOMANIPH
 // don't include iomanip.h becaus it interferes with qt

using namespace covise;

void VrbUiMessageHandler::updateApplicationWindow(const char * cl, int sender, const char * var, covise::TokenBuffer & value)
{
    char * val;
    if (strcmp(cl, "SharedState") != 0)
    {
        value >> val;
       appwin->registry->updateEntry(cl, sender, var, val);
    }
    else
    {
       appwin->registry->updateEntry(cl, sender, var, vrb::tokenBufferToString(std::move(value)).c_str());
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
