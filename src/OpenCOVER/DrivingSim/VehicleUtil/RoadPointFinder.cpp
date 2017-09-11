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
	
	for(int i = 0; i < 12; i++)
	{
		currentRoad[i] = NULL;
		roadPoint[i] = osg::Vec3d();
		currentHeight[i] = 0.0;
		roadHeightIncrement[i] = 0.0;
		roadHeightDifference[i] = 0.0;
		roadHeightDelta[i] = 0.0;
	}
	
	roadHeightIncrementDelta = 0.0005;
	roadHeightIncrementMax = 0.1;
	
	maxRoadDistance = 0.7;
	
	maxHeight = 0.0;
	minHeight = 0.0;
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
		
		double heightGap = std::abs(minHeight - maxHeight);
		for(int i = 0; i < 12; i++)
		{
			//std::cout << "currentRoad " << i << ": " << currentRoad[i] << std::endl;
			//std::cout << "currentLongPos " << i << ": " << currentLongPos[i] << std::endl;
			
			positionMutex.acquire(1000000);
			Vector3D searchInVec(roadPoint[i].x(), roadPoint[i].y(), roadPoint[i].z());//1086.83,1507.96,10.4413);
			positionMutex.release();
			
			currentHeightMutex.acquire(1000000);
			if(currentHeight[i] > maxHeight)
			{
				maxHeight = currentHeight[i];
			}
			if(currentHeight[i] < minHeight)
			{
				minHeight = currentHeight[i];
			}
			double tempCurrentHeight = currentHeight[i];
			currentHeightMutex.release();
			
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
							//std::cout << "road list changed" << std::endl;
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
				
				/*if(heightGap < maxHeightGap)
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
				else
				{
					if(!((roadHeightAverage + roadHeightDelta[i]) > maxHeight || (roadHeightAverage + roadHeightDelta[i]) < minHeight))
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
				}*/
			}
			
			
			/*if(roadList[i].size()==0)
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
					singleRoadSwitch[i] = false;
					std::cout << "road point " << i << " left previous road" << std::endl;
					currentRoadName[i] = RoadSystem::Instance()->getRoadId(roadList[i][0]);
					currentRoadId[i] = 0;
				}
			}
			if(roadList[i].size() == 1)
			{
				double tempHeight = tempCurrentHeight;
				Vector2D v_c = roadList[i][0]->searchPositionNoBorder(searchInVec, -1.0);
				if (!v_c.isNaV())
				{
					RoadPoint point = roadList[i][0]->getRoadPoint(v_c.u(), v_c.v());
					tempHeight = point.z();
				}
				if(!singleRoadSwitch[i])
				{
					if(std::abs(tempCurrentHeight - tempHeight) > 0.005)
					{
						if(tempCurrentHeight > tempHeight)
						{
							if(roadHeightIncrement[i] > -roadHeightIncrementMax)
							{
								roadHeightIncrement[i] = roadHeightIncrement[i] - roadHeightIncrementDelta;
							}
						}
						else if(tempCurrentHeight < tempHeight)
						{
							if(roadHeightIncrement[i] < roadHeightIncrementMax)
							{
								roadHeightIncrement[i] = roadHeightIncrement[i] + roadHeightIncrementDelta;
							}
						}
						tempCurrentHeight = tempCurrentHeight + roadHeightIncrement[i];
						
					}
					else if (std::abs(tempCurrentHeight - tempHeight) > maxRoadDistance)
					{
						if(tempCurrentHeight > tempHeight)
						{
							tempCurrentHeight = tempHeight + maxRoadDistance;
						}
						else if(tempCurrentHeight < tempHeight)
						{
							tempCurrentHeight = tempHeight - maxRoadDistance;
						}
					}
					else
					{
						roadHeightIncrement[i] = 0.0;
						singleRoadSwitch[i] = true;
						//std::cout << "single road switch true for " << i << std::endl;
						tempCurrentHeight = tempHeight;
					}
				}
				else
				{
					tempCurrentHeight = tempHeight;
				}
				roadHeightDifference[i] = std::abs(tempCurrentHeight - tempHeight);
			}
			else if(roadList[i].size() > 1)
			{
				singleRoadSwitch[i] = false;
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
						if(i == 4 || i == 7)
						{
							std::cout << "road height at " << i << " at list position " << j << ": " << point.z() << std::endl; 
						}
					}
					else
					{
						numberInvalidRoads++;
					}
					
				}
				double roadHeightAverage = roadHeightSum / (roadList[i].size() - numberInvalidRoads);
				if(i == 4 || i == 7)
				{
					std::cout << "road height average at " << i << ": " << roadHeightAverage << std::endl; 
				}
				if(std::abs(tempCurrentHeight - roadHeightAverage) > 0.005)
				{
					if(tempCurrentHeight > roadHeightAverage)
					{
						if(roadHeightIncrement[i] > -roadHeightIncrementMax)
						{
							roadHeightIncrement[i] = roadHeightIncrement[i] - roadHeightIncrementDelta;
						}
					}
					else if(tempCurrentHeight < roadHeightAverage)
					{
						if(roadHeightIncrement[i] < roadHeightIncrementMax)
						{
							roadHeightIncrement[i] = roadHeightIncrement[i] + roadHeightIncrementDelta;
						}
					}
				}
				else if (std::abs(tempCurrentHeight - roadHeightAverage) > maxRoadDistance)
				{
					if(tempCurrentHeight > roadHeightAverage)
					{
						tempCurrentHeight = roadHeightAverage + maxRoadDistance;
					}
					else if(tempCurrentHeight < roadHeightAverage)
					{
						tempCurrentHeight = roadHeightAverage - maxRoadDistance;
					}
				}
				else
				{
					roadHeightIncrement[i] = 0.0;
				}
				
				tempCurrentHeight = tempCurrentHeight + roadHeightIncrement[i];
				roadHeightDifference[i] = std::abs(tempCurrentHeight - roadHeightAverage);
			}*/
			currentHeightMutex.acquire(1000000);
			currentHeight[i] = tempCurrentHeight;
			currentHeightMutex.release();
			
		}
		
		rt_task_wait_period(&overruns);
	}
}
