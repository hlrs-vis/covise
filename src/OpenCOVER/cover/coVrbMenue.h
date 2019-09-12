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

}

class VrbMenue: public ui::Owner
{
private:
    ui::Group *sessionGroup;
    ui::Group *ioGroup;
    ui::EditField *newSessionEf;
    ui::Action *newSessionBtn;
    ui::SelectionList *sessionsSl;
    ui::Action *saveBtn;
    ui::SelectionList *loadSL;
    std::vector<std::string> savedRegistries;
    std::vector<vrb::SessionID> availiableSessions;

	vrb::SharedMap<std::string, std::string> test;
    vrb::SharedState<std::vector<float>> test2;
    void saveSession();
    void loadSession(int index);
    void loadSession(const std::string &filename);
    void unloadAll();
    void requestNewSession(const std::string & name);
    void selectSession(int id);
    const std::string noSavedSession = "nothing";
public:
    VrbMenue();
	void initFileMenue();
	void updateState(bool state);
    void updateRegistries(const std::vector<std::string> &registries);
    void updateSessions(const std::vector<vrb::SessionID> &sessions);
    void setCurrentSession(const vrb::SessionID &session);

};


}
#endif
