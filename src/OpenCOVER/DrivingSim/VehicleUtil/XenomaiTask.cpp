/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "XenomaiTask.h"

#include <sys/mman.h>

XenomaiTask::XenomaiTask(const std::string &name, int stacksize, int prio, int mode)
{
    mlockall(MCL_CURRENT | MCL_FUTURE);

    int ret_create = rt_task_create(&rt_task_desc, name.c_str(), stacksize, prio, mode);
    if (ret_create)
    {
        std::cerr << "XenomaiTask::XenomaiTask(): rt_task_create: " << strerror(-ret_create) << std::endl;
    }
}

XenomaiTask::~XenomaiTask()
{
    rt_task_delete(&rt_task_desc);
}

void XenomaiTask::start()
{
    rt_task_start(&rt_task_desc, &XenomaiTask::startTask, static_cast<void *>(this));
}

void XenomaiTask::stop()
{
    rt_task_delete(&rt_task_desc);
}

void XenomaiTask::suspend()
{
    rt_task_suspend(&rt_task_desc);
}

void XenomaiTask::resume()
{
    rt_task_resume(&rt_task_desc);
}

void XenomaiTask::set_periodic(RTIME nanoseconds)
{
    int ecode = rt_task_set_periodic(&rt_task_desc, TM_NOW, (nanoseconds));
    std::cerr << "XenomaiTask::set_periodic(): " << strerror(-ecode) << std::endl;
}

int XenomaiTask::inquire(RT_TASK_INFO &info)
{
    return rt_task_inquire(&rt_task_desc, &info);
}

void XenomaiTask::startTask(void *data)
{
    XenomaiTask *task = static_cast<XenomaiTask *>(data);
    task->run();
}
