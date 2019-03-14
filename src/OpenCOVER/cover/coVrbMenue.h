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

namespace opencover {
namespace ui
{
class Action;
class EditField;
class Menu;
class Owner;
class Group;
class SelectionList;

}

class VrbMenue
{
private:
    ui::Owner *m_owner;
    ui::Menu *menue;
    std::shared_ptr<ui::EditField> newSessionEf;
    std::shared_ptr<ui::Action> newSessionBtn;
    std::shared_ptr<ui::SelectionList> SessionsSl;
    std::unique_ptr<ui::Action> saveBtn;
    std::unique_ptr<ui::SelectionList> loadSL;
    std::vector<std::string> savedRegistries;
    std::vector<vrb::SessionID> availiableSessions;
    void init();
    void saveSession();
    void loadSession(const std::string &filename);
    void unloadAll();
    const std::string noSavedSession = "nothing";
public:
    VrbMenue(ui::Owner *owner);
    void updateState(bool state);
    void updateRegistries(const std::vector<std::string> &registries);
    void updateSessions(const std::vector<vrb::SessionID> &sessions);
    void setCurrentSession(const vrb::SessionID &session);
    
};


}
#endif
