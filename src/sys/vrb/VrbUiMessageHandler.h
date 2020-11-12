/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef VRBUIMESAGEHANDLER_H
#define VRBUIMESAGEHANDLER_H
#endif // !VRBUIMESAGEHANDLER_H

#include <vrb/server/VrbMessageHandler.h>

namespace covise
{
class Message;
class DataHandle;
}
class VrbUiMessageHandler : public vrb::VrbMessageHandler
{
public:
    using vrb::VrbMessageHandler::VrbMessageHandler;
    void updateApplicationWindow(const std::string& cl, int sender, const std::string& var, const covise::DataHandle& value) override;
    void removeEntryFromApplicationWindow(const std::string& cl, int sender, const std::string& var) override;
    void removeEntriesFromApplicationWindow(int sender) override;
    ///get the client corresponding to con and change its QSocketNotifier state; Return true if client exists
    bool setClientNotifier(covise::Connection *conn, bool state);
};