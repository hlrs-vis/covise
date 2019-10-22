/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef NVTOP_PLUGIN_H
#define NVTOP_PLUGIN_H

#include <cover/coVRPlugin.h>

#include <thread>
#include <memory>
#include <mutex>

struct device_info;

class NvTop: public opencover::coVRPlugin
{
public:
    NvTop();
    ~NvTop();

    bool init() override;
    void preFrame() override;


private:
    unsigned m_numdevs = 0;
    struct device_info *m_devinfos = nullptr, *m_threadDevInfos = nullptr;

    std::unique_ptr<std::thread> m_thread;
    std::mutex m_mutex;
    bool m_needThread = false;

    void stopThread();
};
#endif
