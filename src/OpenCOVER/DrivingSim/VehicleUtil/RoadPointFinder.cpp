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
	
	loadingRoad = true;
	
	for(int i = 0; i < 12; i++)
	{
		currentRoad[i] = NULL;
		roadPoint[i] = osg::Vec3d();
		currentHeight[i] = 0.0;
		roadHeightIncrement[i] = 0.0;
		roadHeightDifference[i] = 0.0;
		roadHeightDelta[i] = 0.0;
		leftRoadSwitch[i] = false;
	}
	
	roadHeightIncrementDelta = 0.0005;
	roadHeightIncrementMax = 0.1;
	
	maxRoadDistance = 0.7;
	
	maxHeight = -10000000000.0;
	minHeight = 10000000000.0;
	maxHeightGap = 1.5;
	
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
		std::cout << "loadingRoad: " << loadingRoad << std::endl;
	}
	
	std::cerr << "RoadPointFinder is done waiting" << std::endl;
	
	for(int i = 0; i < 12; i++)
	{
		std::cout << "iCounter: " << i << std::endl;
		positionMutex.acquire(1000000);
		Vector3D searchInVec(roadPoint[i].x(), roadPoint[i].y(), roadPoint[i].z());//1086.83,1507.96,10.4413);
		positionMutex.release();
		
		std::cout << "searchInVec x: " << searchInVec.x() << " y: " << searchInVec.y() << " z: " << searchInVec.z() << std::endl;
		
		currentHeightMutex.acquire(1000000);
		double tempCurrentHeight = currentHeight[i];
		currentHeightMutex.release();
		
		//list system
		std::vector<Road*> tempRoadList;
		std::vector<Road*> oldRoadList;
		tempRoadList = RoadSystem::Instance()->searchPositionList(searchInVec);
		
		bool roadListChanged = false;
		
		if(tempRoadList.size() != 0)
		{
			std::vector<Road*> oldRoadList;
			oldRoadList = roadList[i];
			roadList[i] = tempRoadList;
			
			if(oldRoadList.size() == roadList[i].size())
			{
				for(int j = 0; j < roadList[i].size(); j++)
				{
					bool roadFound = false;
					std::string tempRoadName = RoadSystem::Instance()->getRoadId(roadList[i][j]);
					for(int k = 0; k < oldRoadList.size(); k++)
					{
						if(tempRoadName.compare(RoadSystem::Instance()->getRoadId(oldRoadList[k])) == 0)
						{
							roadFound = true;
							break;
						}
					}
					if(roadFound == false)
					{
						roadListChanged = true;
						break;
					}
				}
			}
			else
			{
				roadListChanged = true;
			}
		}
		else
		{
			singleRoadSwitch[i] = false;
		}
		
		double roadHeightAverage = tempCurrentHeight;
		if(roadList[i].size() > 0)
		{
			if(roadList[i].size() > 1)
			{
				int numberInvalidRoads = 0;
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
					else
					{
						numberInvalidRoads++;
					}
					
				}
				if(!(roadList[i].size() - numberInvalidRoads == 0))
				{
					roadHeightAverage = roadHeightSum / (roadList[i].size() - numberInvalidRoads);
				}
			}
			else
			{
				Vector2D v_c = roadList[i][0]->searchPositionNoBorder(searchInVec, -1);
				//std::cout << "v_c: " << v_c.u() << " " << v_c.v() << std::endl;
				if (!v_c.isNaV())
				{
					RoadPoint point = roadList[i][0]->getRoadPoint(v_c.u(), v_c.v());
					roadHeightAverage = point.z();
				}
			}
			
			if(roadListChanged)
			{
				roadHeightDelta[i] = tempCurrentHeight - roadHeightAverage;
				roadListChanged = false;
			}
			
			tempCurrentHeight = roadHeightAverage + roadHeightDelta[i];
				
			if(roadHeightDelta[i] > 0)
			{
				roadHeightDelta[i] = roadHeightDelta[i] - roadHeightIncrementDelta;
			}
			if(roadHeightDelta[i] < 0)
			{
				roadHeightDelta[i] = roadHeightDelta[i] + roadHeightIncrementDelta;
			}
			if(std::abs(roadHeightDelta[i]) < 0.001)
			{
				roadHeightDelta[i] = 0.0;
			}
			
			std::cout << "tempCurrentHeight " << i << ": " << tempCurrentHeight << std::endl;
			
			
		}
		
		currentHeightMutex.acquire(1000000);
		currentHeight[i] = tempCurrentHeight;
		currentHeightMutex.release();
		
	}
	
	while (runTask)
    {
		if (overruns != 0)
		{
			std::cerr << "RoadPointFinder::run(): overruns: " << overruns << std::endl;
			overruns=0;
		}
		
		//std::cout << "RoadPointFinder running" << std::endl;
		
		maxHeight = -10000000000.0;
		minHeight = 10000000000.0;
		for(int i = 0; i < 12; i++)
		{
			currentHeightMutex.acquire(1000000);
			if(currentHeight[i] > maxHeight)
			{
				maxHeight = currentHeight[i];
			}
			if(currentHeight[i] < minHeight)
			{
				minHeight = currentHeight[i];
			}
			currentHeightMutex.release();
		}
		
		double heightGap = std::abs(minHeight - maxHeight);
		std::cout << "maxHeight: " << maxHeight << " minHeight: " << minHeight << " height gap: " << heightGap << std::endl;
		for(int i = 0; i < 12; i++)
		{
			//std::cout << "currentRoad " << i << ": " << currentRoad[i] << std::endl;
			//std::cout << "currentLongPos " << i << ": " << currentLongPos[i] << std::endl;
			
			positionMutex.acquire(1000000);
			Vector3D searchInVec(roadPoint[i].x(), roadPoint[i].y(), roadPoint[i].z());//1086.83,1507.96,10.4413);
			positionMutex.release();
			
			currentHeightMutex.acquire(1000000);
			double tempCurrentHeight = currentHeight[i];
			currentHeightMutex.release();
			
			//list system
			std::vector<Road*> tempRoadList;
			std::vector<Road*> oldRoadList;
			tempRoadList = RoadSystem::Instance()->searchPositionList(searchInVec);
			
			bool roadListChanged = false;
			
			if(tempRoadList.size() != 0)
			{
				/*for(int j = 0; j < tempRoadList.size(); j++)
				{
					std::cout << "road id at position: " << j << "; " << RoadSystem::Instance()->getRoadId(tempRoadList[j]) << std::endl;
					
				}*/
				
				std::vector<Road*> oldRoadList;
				oldRoadList = roadList[i];
				roadList[i] = tempRoadList;
				
				if(oldRoadList.size() == roadList[i].size())
				{
					for(int j = 0; j < roadList[i].size(); j++)
					{
						bool roadFound = false;
						std::string tempRoadName = RoadSystem::Instance()->getRoadId(roadList[i][j]);
						for(int k = 0; k < oldRoadList.size(); k++)
						{
							if(tempRoadName.compare(RoadSystem::Instance()->getRoadId(oldRoadList[k])) == 0)
							{
								roadFound = true;
								break;
							}
						}
						if(roadFound == false)
						{
							roadListChanged = true;
							//std::cout << "road list changed" << std::endl;
							break;
						}
					}
				}
				else
				{
					roadListChanged = true;
				}
				if(leftRoadSwitch[i] == true)
				{
					leftRoadSwitch[i] = false;
					roadListChanged = true;
				}
			}
			else
			{
				singleRoadSwitch[i] = false;
				leftRoadSwitch[i] = true;
				//std::cout << "single road switch false" << std::endl;
			}
			
			//std::cout << "road list at " << i << ": " << roadList[i].size() << std::endl;
			//std::cout << "search in vector " << i << ": " << roadPoint[i].x() << " " << roadPoint[i].y() << " " << roadPoint[i].z() << std::endl;
			
			double roadHeightAverage = tempCurrentHeight;
			if(roadList[i].size() > 0)
			{
				if(roadList[i].size() > 1)
				{
					int numberInvalidRoads = 0;
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
						else
						{
							numberInvalidRoads++;
						}
						
					}
					if(!(roadList[i].size() - numberInvalidRoads == 0))
					{
						roadHeightAverage = roadHeightSum / (roadList[i].size() - numberInvalidRoads);
					}
				}
				else
				{
					Vector2D v_c = roadList[i][0]->searchPositionNoBorder(searchInVec, -1);
					//std::cout << "v_c: " << v_c.u() << " " << v_c.v() << std::endl;
					if (!v_c.isNaV())
					{
						RoadPoint point = roadList[i][0]->getRoadPoint(v_c.u(), v_c.v());
						roadHeightAverage = point.z();
					}
				}
				
				if(roadListChanged)
				{
					roadHeightDelta[i] = tempCurrentHeight - roadHeightAverage;
					roadListChanged = false;
				}
				
				std::cout << "roadHeightAverage at position " << i << ": " << roadHeightAverage << std::endl;
				
				if(heightGap < maxHeightGap)
				{
					if(!((roadHeightAverage + roadHeightDelta[i] > minHeight + maxHeightGap) || (roadHeightAverage + roadHeightDelta[i] < maxHeight - maxHeightGap)))
					{	
						tempCurrentHeight = roadHeightAverage + roadHeightDelta[i];
						
						if(roadHeightDelta[i] > 0)
						{
							roadHeightDelta[i] = roadHeightDelta[i] - roadHeightIncrementDelta;
						}
						if(roadHeightDelta[i] < 0)
						{
							roadHeightDelta[i] = roadHeightDelta[i] + roadHeightIncrementDelta;
						}
						if(std::abs(roadHeightDelta[i]) < 0.001)
						{
							roadHeightDelta[i] = 0.0;
						}
					}
				}
				else
				{
					if(!((roadHeightAverage + roadHeightDelta[i]) > maxHeight || (roadHeightAverage + roadHeightDelta[i]) < minHeight))
					{
						if(!((roadHeightAverage + roadHeightDelta[i] > minHeight + maxHeightGap) || (roadHeightAverage + roadHeightDelta[i] < maxHeight - maxHeightGap)))
						{
							tempCurrentHeight = roadHeightAverage + roadHeightDelta[i];
							
							if(roadHeightDelta[i] > 0)
							{
								roadHeightDelta[i] = roadHeightDelta[i] - roadHeightIncrementDelta;
							}
							if(roadHeightDelta[i] < 0)
							{
								roadHeightDelta[i] = roadHeightDelta[i] + roadHeightIncrementDelta;
							}
							if(std::abs(roadHeightDelta[i]) < 0.001)
							{
								roadHeightDelta[i] = 0.0;
							}
						}
					}
				}
			}
			
			std::cout << "current height at position " << i << ": " << tempCurrentHeight << std::endl;
			
			currentHeightMutex.acquire(1000000);
			currentHeight[i] = tempCurrentHeight;
			currentHeightMutex.release();
			
		}
		
		rt_task_wait_period(&overruns);
	}
}
