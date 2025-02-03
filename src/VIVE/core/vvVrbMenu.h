/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include "ui/Owner.h"

#include <vrb/SessionID.h>

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace vrb
{
class SessionID;
}
namespace vive {
namespace ui
{
class Action;
class EditField;
class Menu;
class Group;
class SelectionList;
class Slider;
class FileBrowser;
}

class vvMessageSender;
class VrbMenu;

class VrbMenu : public ui::Owner
{
private:
    std::unique_ptr<vvMessageSender> sender;
    const std::string noSavedSession = "nothing";
    ui::Group *m_sessionGroup;
    ui::Group *m_ioGroup;
    ui::EditField *m_newSessionEf;
    ui::Action *m_newSessionBtn;
    ui::SelectionList *m_sessionsSl, *m_remoteLauncher;
    ui::FileBrowser *m_saveSession, *m_loadSession;
    std::vector<std::string> m_savedRegistries;
    std::vector<vrb::SessionID> m_availiableSessions;
    std::vector<std::function<void()>> m_onSessionChangedCallbacks;
    void saveSession(const std::string &file);
    void loadSession(const std::string &filename);
    void requestNewSession(const std::string & name);
    void selectSession(int id);
    void lauchRemotePartner(int id);

public:
    VrbMenu();
	void initFileMenu();
	void updateState(bool state);
    void updateSessions(const std::vector<vrb::SessionID> &sessions);
    void updateRemoteLauncher();
    void setCurrentSession(const vrb::SessionID &session);
    ~VrbMenu() = default;
};
    int getRemoteLauncherClientID(int index);


}
