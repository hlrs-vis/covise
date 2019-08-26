/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef VRBUIMESAGEHANDLER_H
#define VRBUIMESAGEHANDLER_H
#endif // !VRBUIMESAGEHANDLER_H

#include <vrbserver/VrbMessageHandler.h>

namespace covise
{
class Message;
class DataHandle;
}
class VrbUiMessageHandler : public vrb::VrbMessageHandler
{
public:
    using vrb::VrbMessageHandler::VrbMessageHandler;
    void updateApplicationWindow(const char *cl, int sender, const char *var, const covise::DataHandle& value) override;
    void removeEntryFromApplicationWindow(const char *cl, int sender, const char *var) override;
    void removeEntriesFromApplicationWindow(int sender) override;
    ///get the client corresponding to con and change its QSocketNotifier state; Return true if client exists
    bool setClientNotifier(covise::Connection *conn, bool state);
};