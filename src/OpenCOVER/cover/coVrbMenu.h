/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVRBMENU_H
#define COVRBMENU_H


#include "ui/Owner.h"

#include <vrb/SessionID.h>
#include <vrb/remoteLauncher/VrbRemoteLauncher.h>

#include <string>
#include <vector>
namespace vrb
{
class SessionID;
}
namespace opencover {
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
class VrbMenu;

class VrbMenu : public ui::Owner
{
private:
    const std::string noSavedSession = "nothing";
    ui::Group *m_sessionGroup;
    ui::Group *m_ioGroup;
    ui::EditField *m_newSessionEf;
    ui::Action *m_newSessionBtn;
    ui::SelectionList *m_sessionsSl, *m_remoteLauncher;
    ui::FileBrowser *m_saveSession, *m_loadSession;
    std::vector<std::string> m_savedRegistries;
    std::vector<vrb::SessionID> m_availiableSessions;

    void saveSession(const std::string &file);
    void loadSession(const std::string &filename);
    void requestNewSession(const std::string & name);
    void selectSession(int id);

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
#endif
