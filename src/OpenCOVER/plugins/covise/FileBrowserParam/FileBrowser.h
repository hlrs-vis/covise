/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FILEBROWSER_PLUGIN_H
#define FILEBROWSER_PLUGIN_H

//**************************************************************************
//
// * Description    : Plugin for handling one FileBrowser parameter
//
// * Author  : Martin Aumueller
//
// **************************************************************************

#include <cover/coVRPlugin.h>
#include <CovisePluginUtil/coBaseCoviseInteractor.h>
#include <cover/ui/Owner.h>

namespace opencover {
namespace ui {
class ButtonGroup;
class Menu;
}
}

using namespace covise;
using namespace opencover;

class FileBrowser : public coVRPlugin, public ui::Owner
{
public:
    // Constructor
    FileBrowser();

    // Destructor
    ~FileBrowser();

    bool init();

    static FileBrowser *instance();

    // this will be called if a COVISE object has to be removed
    void removeObject(const char *objName, bool replace);

    // this will be called when a COVISE object with a feedback object arrives
    void newInteractor(const RenderObject *ro, coInteractor *inter);

private:
    // class functions
    void addMenuEntry(); // Add a menu to main pinboard
    void readDirectory();
    void removeMenuEntry(); // And remove it...

#if 0
    // Callback function for mouse clicks
    static void fileSelection(void *, buttonSpecCell *);
#endif

    ui::ButtonGroup *m_buttonGroup = nullptr;
    ui::Menu *m_menu = nullptr;

    coBaseCoviseInteractor *m_interactor;
    std::string m_interactorObjName;
    std::string m_moduleName;
    std::string m_paramName;

    std::string m_basedir;
    std::list<std::string> m_files;

    static FileBrowser *s_instance;
};

#endif
