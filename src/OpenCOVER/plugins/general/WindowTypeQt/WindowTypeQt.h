/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WINDOW_TYPE_QT
#define WINDOW_TYPE_QT

#include <cover/coVRPlugin.h>
#include <vector>

class QMainWindow;
class QtOsgWidget;
class QAction;
class QDialog;

namespace opencover {
class QtMainWindow;
namespace ui {
class QtView;
}
}

class WindowTypeQtPlugin : public opencover::coVRPlugin
{
public:
    WindowTypeQtPlugin();
    ~WindowTypeQtPlugin();
    bool destroy() override;
    bool update() override;

    bool windowCreate(int num) override;
    void windowCheckEvents(int num) override;
    void windowUpdateContents(int num) override;
    void windowDestroy(int num) override;

private:
    void aboutCover() const;
    struct WindowData
    {
        int index = -1;
        opencover::QtMainWindow *window = nullptr;
        QtOsgWidget *widget = nullptr;
        QAction *toggleMenu = nullptr;
        std::vector<opencover::ui::QtView *> view;
    };
    std::map<int, WindowData> m_windows;
    bool m_update = true;
    QDialog *m_keyboardHelp = nullptr;
};
#endif
