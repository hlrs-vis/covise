/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */
#ifndef VRBUIMESAGEHANDLER_H
#define VRBUIMESAGEHANDLER_H
#endif // !VRBUIMESAGEHANDLER_H

#include <vrb/server/VrbMessageHandler.h>
#include <QSocketNotifier>
namespace covise
{
class Message;
class DataHandle;
}

struct UiConnectionDetails : vrb::ConnectionDetails{
  typedef std::unique_ptr<UiConnectionDetails> ptr;
  std::unique_ptr<QSocketNotifier> notifier;
  UiConnectionDetails() = default;
  UiConnectionDetails(UiConnectionDetails &) = delete;
  UiConnectionDetails &operator=(UiConnectionDetails &) = delete;
  UiConnectionDetails(UiConnectionDetails &&) = default;
  UiConnectionDetails &operator=(UiConnectionDetails &&) = default;
};

class VrbUiMessageHandler : public vrb::VrbMessageHandler
{
public:
    using vrb::VrbMessageHandler::VrbMessageHandler;
    void updateApplicationWindow(const std::string& cl, int sender, const std::string& var, const covise::DataHandle& value) override;
    void removeEntryFromApplicationWindow(const std::string& cl, int sender, const std::string& var) override;
    void removeEntriesFromApplicationWindow(int sender) override;
    ///get the client corresponding to con and change its QSocketNotifier state; Return true if client exists
    bool setClientNotifier(const covise::Connection *conn, bool state);
    vrb::VRBSClient *createNewClient(vrb::ConnectionDetails::ptr &&, covise::TokenBuffer &tb) override;

};