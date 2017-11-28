/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WINDOW_TYPE_MESA
#define WINDOW_TYPE_MESA

#include <cover/coVRPlugin.h>
#include <GL/osmesa.h>

class WindowData
{
    public:
    OSMesaContext context;
    char *buffer;
    int index;
};

class WindowTypeMesaPlugin : public opencover::coVRPlugin
{
public:
    WindowTypeMesaPlugin();
    ~WindowTypeMesaPlugin();
    bool destroy() override;
    bool update() override;

    bool windowCreate(int num) override;
    void windowCheckEvents(int num) override;
    void windowUpdateContents(int num) override;
    void windowDestroy(int num) override;

private:
    std::map<int, WindowData> m_windows;
};
#endif
