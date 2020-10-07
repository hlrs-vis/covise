/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVRBMENUE_H
#define COVRBMENUE_H

#include <string>
#include <set>
#include <vector>
#include <memory>
#include <vrbclient/SessionID.h>
#include "ui/Owner.h"
#include <vrbclient/SharedState.h>
//test
#include <vrbclient/SharedState.h>
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

class VrbMenue: public ui::Owner
{
private:
    ui::Group *m_sessionGroup;
    ui::Group *m_ioGroup;
    ui::EditField *m_newSessionEf;
    ui::Action *m_newSessionBtn;
    ui::SelectionList *m_sessionsSl;
    ui::FileBrowser *m_saveSession, *m_loadSession;
    std::vector<std::string> m_savedRegistries;
    std::vector<vrb::SessionID> m_availiableSessions;

    void saveSession(const std::string &file);
    void loadSession(const std::string &filename);
    void requestNewSession(const std::string & name);
    void selectSession(int id);
    const std::string noSavedSession = "nothing";
public:
    VrbMenue();
	void initFileMenue();
	void updateState(bool state);
    void updateSessions(const std::vector<vrb::SessionID> &sessions);
    void setCurrentSession(const vrb::SessionID &session);

};


}
#endif
