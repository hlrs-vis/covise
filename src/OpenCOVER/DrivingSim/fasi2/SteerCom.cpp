#include "SteerCom.h"
#include <unistd.h>

SteerCom::SteerCom()
#ifdef MERCURY
    : XenomaiTask::XenomaiTask("SteerComTask", 0, 99, 0)
#else
    : XenomaiTask::XenomaiTask("SteerComTask", 0, 99, T_FPU | T_CPU(5))
#endif
{
	runTask = true;
	taskFinished = false;
	
	current = 0;
	steerWheelAngle = 0;
	
	steerCon = new CanOpenController("can1");
    steerWheel = new XenomaiSteeringWheel(*steerCon, 1);
	
	start();
}

SteerCom::~SteerCom()
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
    steerWheel->shutdown();
    
    delete steerWheel;
    delete steerCon;
}

void SteerCom::run()
{
	std::cerr << "Starting steering wheel communication..." << std::endl;
	rt_task_set_periodic(NULL, TM_NOW, rt_timer_ns2ticks(1000000)); //unterschied set_periodic()????
	
	std::cerr << "steering communication innitiates steerWheel..." << std::endl;
    steerWheel->init();
	
	while (runTask)
    {
		if (overruns != 0)
		{
			std::cerr << "SteerCom::run(): overruns: " << overruns << std::endl;
			overruns=0;
		}
		
		current = 0;
		std::cout << "steer com current:" << current << " steer wheel angle: " << steerWheelAngle << std::endl;
		
		int32_t steerPosition;
		steerCon->sendSync();
		steerCon->recvPDO(1);
		steerPosition = steerWheel->getPosition();
		steerWheelAngle = (double)steerPosition / (double)steerWheel->countsPerTurn;
		
		steerWheel->setCurrent(current);
		steerCon->sendPDO();
		
		rt_task_wait_period(&overruns);
	}
	
	steerWheel->setCurrent(0);
}

void SteerCom::setCurrent(double inCurrent)
{
	//current = inCurrent;
}

double SteerCom::getSteeringWheelAngle()
{
	
	return steerWheelAngle;
}

void SteerCom::centerCall()
{
	steerWheel->center();
}

	