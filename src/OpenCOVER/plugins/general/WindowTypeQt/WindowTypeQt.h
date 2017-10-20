/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WINDOW_TYPE_QT
#define WINDOW_TYPE_QT

#include <cover/coVRPlugin.h>

class QMainWindow;
class QtOsgWidget;

namespace opencover {
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

    bool windowCreate(int num) override;
    void windowCheckEvents(int num) override;
    void windowUpdateContents(int num) override;
    void windowDestroy(int num) override;

private:
    struct WindowData
    {
        int index = -1;
        QMainWindow *window = nullptr;
        QtOsgWidget *widget = nullptr;
        opencover::ui::QtView *view = nullptr;
    };
    std::map<int, WindowData> m_windows;
};
#endif
