#include "RoadPointFinder.h"

#include <unistd.h>

RoadPointFinder::RoadPointFinder()
#ifdef MERCURY
    : XenomaiTask::XenomaiTask("RoadPointFinderTask", 0, 99, 0)
#else
    : XenomaiTask::XenomaiTask("RoadPointFinderTask", 0, 99, T_FPU | T_CPU(5))
#endif
	, roadMutex("rpf_road_mutex")
	, positionMutex("rpf_position_mutex")
	, longPosMutex("rpf_longpos_mutex")
	, overruns(0)
{
	runTask = true;
	taskFinished = false;
	
	for(int i = 0; i < 12; i++)
	{
		currentRoad[i] = NULL;
		roadPoint[i] = osg::Vec3d();
	}
	
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
	std::cout << "Starting RoadPointFinder..." << std::endl;
	rt_task_set_periodic(NULL, TM_NOW, rt_timer_ns2ticks(3000000));
	
	while(loadingRoad)
	{
	}
	
	std::cerr << "RoadPointFinder is done waiting" << std::endl;
	
	while (runTask)
    {
		if (overruns != 0)
		{
			std::cerr << "RoadPointFinder::run(): overruns: " << overruns << std::endl;
			overruns=0;
		}
		
		//std::cout << "RoadPointFinder running" << std::endl;
		
		for(int i = 0; i < 12; i++)
		{
			//std::cout << "currentRoad " << i << ": " << currentRoad[i] << std::endl;
			//std::cout << "currentLongPos " << i << ": " << currentLongPos[i] << std::endl;
			
			positionMutex.acquire(1000000);
			Vector3D searchInVec(roadPoint[i].x(), roadPoint[i].y(), roadPoint[i].z());//1086.83,1507.96,10.4413);
			positionMutex.release();
			
			//std::cout << "search in vector " << i << ": " << roadPoint[i].x() << roadPoint[i].y() << roadPoint[i].z() << std::endl;
			
			roadMutex.acquire(1000000);
			longPosMutex.acquire(1000000);
			//std::cout << "fl1 in" << std:: endl << "pointx" << tempVec.x() << std::endl << "pointy" << tempVec.y() << std::endl << "pointz" << tempVec.z() << std::endl;
			Vector2D searchOutVec = RoadSystem::Instance()->searchPosition(searchInVec, currentRoad[i], currentLongPos[i]);
			
			
			
			//std::cout << "searchOutVec " << i << ": " << searchOutVec.x() << " " << searchOutVec.y() << std::endl;
			
			//std::cout << "fl2 out" << std::endl << "pointx" << searchOutVec.x() << std::endl << "pointy" << searchOutVec.y() << std::endl;
			currentLongPos[i] = searchOutVec.x();
			longPosMutex.release();
			
			/*RoadPoint point = currentRoad[i]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
			
			std::cout << "pointx" << point.x() << std::endl << "pointy" << point.y() << std::endl << "pointz" << point.z() << std::endl;*/
			
			roadMutex.release();
			/*if (isnan(point.x()))
			{
				std::cout << "tire point " << i << " left road!" << std::endl;
			}*/
			
		}
		
		rt_task_wait_period(&overruns);
	}
}
