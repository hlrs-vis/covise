#include "RoadPointFinder.h"
#include <unistd.h>

RoadPointFinder::RoadPointFinder()
#ifdef MERCURY
    : XenomaiTask::XenomaiTask("RoadPointFinderTask", 0, 99, 0)
#else
    : XenomaiTask::XenomaiTask("RoadPointFinderTask", 0, 99, T_FPU | T_CPU(5))
#endif
{
	runTask = true;
	taskFinished = false;
	
	start();
}

RoadPointFinder::~RoadPointFinder()
{
    RT_TASK_INFO info;
    inquire(info);

#ifdef MERCURY
    if (info.stat.status & __THREAD_S_STARTED)
#else
    if (info.status & T_STARTED)
#endif

    {
        runTask = false;
        while (!taskFinished)
        {
            usleep(100000);
        }
    }
}

void RoadPointFinder::run()
{
	std::cerr << "Starting RoadPointFinder..." << std::endl;
	rt_task_set_periodic(NULL, TM_NOW, rt_timer_ns2ticks(1000000));
	
	
	while (runTask)
    {
		if (overruns != 0)
		{
			std::cerr << "SteerCom::run(): overruns: " << overruns << std::endl;
			overruns=0;
		}
	}
}
