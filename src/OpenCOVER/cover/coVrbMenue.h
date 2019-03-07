/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVRBMENUE_H
#define COVRBMENUE_H

#include <string>
#include <set>

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
    std::set<std::string> savedRegistries;
    std::set<std::vector> availiableSessions;

    void init();
    void removeVRB_UI();
    void saveSession();
    void loadSession(std::string &filename);
    void unloadAll();

public:
    VrbMenue(ui::Owner *owner);
    void updateState(bool state);
    void updateRegistries(std::vector<std::string> &registries);
};


}
#endif