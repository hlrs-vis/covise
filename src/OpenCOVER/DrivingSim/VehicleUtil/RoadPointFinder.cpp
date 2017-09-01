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
	, currentHeightMutex("rpf_currentHeight_mutex")
	, overruns(0)
{
	runTask = true;
	taskFinished = false;
	
	roadHeightIncrement = 0.003;
	
	for(int i = 0; i < 12; i++)
	{
		currentRoad[i] = NULL;
		roadPoint[i] = osg::Vec3d();
		currentHeight[i] = 0;
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
			
			/*roadMutex.acquire(1000000);
			longPosMutex.acquire(1000000);
			//std::cout << "fl1 in" << std:: endl << "pointx" << tempVec.x() << std::endl << "pointy" << tempVec.y() << std::endl << "pointz" << tempVec.z() << std::endl;
			Vector2D searchOutVec = RoadSystem::Instance()->searchPosition(searchInVec, currentRoad[i], currentLongPos[i]);*/
			
			
			
			//std::cout << "searchOutVec " << i << ": " << searchOutVec.x() << " " << searchOutVec.y() << std::endl;
			
			//std::cout << "fl2 out" << std::endl << "pointx" << searchOutVec.x() << std::endl << "pointy" << searchOutVec.y() << std::endl;
			/*currentLongPos[i] = searchOutVec.x();
			longPosMutex.release();*/
			
			/*RoadPoint point = currentRoad[i]->getRoadPoint(searchOutVec.x(), searchOutVec.y());
			
			std::cout << "pointx" << point.x() << std::endl << "pointy" << point.y() << std::endl << "pointz" << point.z() << std::endl;*/
			
			//roadMutex.release();
			/*if (isnan(point.x()))
			{
				std::cout << "tire point " << i << " left road!" << std::endl;
			}*/
			
			//list system
			roadList[i] = RoadSystem::Instance()->searchPositionList(searchInVec);
			
			//std::cout << "road list at " << i << ": " << roadList[i].size() << std::endl;
			//std::cout << "search in vector " << i << ": " << roadPoint[i].x() << " " << roadPoint[i].y() << " " << roadPoint[i].z() << std::endl;
			
			if(roadList[i].size()==0)
			{
				currentRoadName[i] = "asdasdasdasdafergbdv;bonesafae";
				singleRoadSwitch[i] = false;
			}
			else
			{
				bool stillOnRoad = false;
				for(int j = 0; j < roadList[i].size(); j++)
				{
					if(currentRoadName[i].compare(RoadSystem::Instance()->getRoadId(roadList[i][j])) == 0)
					{
						stillOnRoad = true;
						currentRoadId[i] = i;
					}
				}
				if(stillOnRoad == false) 
				{
					std::cout << "road point " << i << " left previous road" << std::endl;
					currentRoadName[i] = RoadSystem::Instance()->getRoadId(roadList[i][0]);
					currentRoadId[i] = 0;
				}
			}
			if(roadList[i].size() == 1)
			{
				currentHeightMutex.acquire(1000000);
				double tempHeight = currentHeight[i];
				Vector2D v_c = roadList[i][0]->searchPositionNoBorder(searchInVec, -1.0);
				if (!v_c.isNaV())
				{
					RoadPoint point = roadList[i][0]->getRoadPoint(v_c.u(), v_c.v());
					tempHeight = point.z();
				}
				if(!singleRoadSwitch[i])
				{
					
					if(std::abs(currentHeight[i] - tempHeight) > 0.005)
					{
						if(currentHeight[i] > tempHeight)
						{
							currentHeight[i] = currentHeight[i] - roadHeightIncrement;
						}
						else if(currentHeight[i] < tempHeight)
						{
							currentHeight[i] = currentHeight[i] + roadHeightIncrement;
						}
					}
					else
					{
						singleRoadSwitch[i] = true;
						std::cout << "single road switch true for " << i << std::endl;
						currentHeight[i] = tempHeight;
					}
				}
				else
				{
					currentHeight[i] = tempHeight;
				}
				currentHeightMutex.release();
			}
			else if(roadList[i].size() > 1)
			{
				singleRoadSwitch[i] = false;
				
				//calculate average height
				double roadHeightSum = 0;
				for(int j = 0; j < roadList[i].size(); j++)
				{
					Vector2D v_c = roadList[i][j]->searchPosition(searchInVec, 0);
					if (!v_c.isNaV())
					{
						RoadPoint point = roadList[i][j]->getRoadPoint(v_c.u(), v_c.v());
						roadHeightSum = roadHeightSum + point.z();
					}
				}
				currentHeightMutex.acquire(1000000);
				double roadHeightAverage = roadHeightSum / roadList[i].size();
				if(currentHeight[i] > roadHeightAverage)
				{
					currentHeight[i] = currentHeight[i] - roadHeightIncrement;
				}
				else if(currentHeight[i] < roadHeightAverage)
				{
					currentHeight[i] = currentHeight[i] + roadHeightIncrement;
				}
				currentHeightMutex.release();
			}
			
		}
		
		rt_task_wait_period(&overruns);
	}
}
