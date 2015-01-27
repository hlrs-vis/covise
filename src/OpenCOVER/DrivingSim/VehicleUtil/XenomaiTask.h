/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __XenomaiTask_h
#define __XenomaiTask_h

#include <iostream>

#include <native/task.h>

class XenomaiTask
{
public:
    XenomaiTask(const std::string & = "noname", int = 0, int = 99, int = 0);
    virtual ~XenomaiTask();

    void start();
    void stop();

    void suspend();
    void resume();

    void set_periodic(RTIME);

    int inquire(RT_TASK_INFO &);

protected:
    static void startTask(void *);

    virtual void run() = 0;

    RT_TASK rt_task_desc;
};

#endif
