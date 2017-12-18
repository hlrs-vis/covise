#ifndef __RoadPointFinder_h
#define __RoadPointFinder_h

#include "XenomaiTask.h"
#include "XenomaiMutex.h"

#include "RoadSystem/RoadSystem.h"

class VEHICLEUTILEXPORT RoadPointFinder : public XenomaiTask
{
public:
	RoadPointFinder();
	~RoadPointFinder();
	
	void setRoad(Road *&road, int contactPointNumber);
	Road* getRoad(int contactPointNumber);
	
	void setPoint(osg::Vec3d inVector, int contactPointNumber);
	
	double getHeight(int contactPointNumber);
	void setHeight(double height, int contactPointNumber);
	
	double getLongPos(int contactPointNumber);
	void setLongPos(double inLongPos, int contactPointNumber);
	
	void checkLoadingRoads(bool loadingState);
	
	XenomaiMutex roadMutex;
	XenomaiMutex positionMutex;
	XenomaiMutex longPosMutex;
	XenomaiMutex currentHeightMutex;
	
	XenomaiMutex &getRoadMutex()
    {
        return roadMutex;
    }
    
    XenomaiMutex &getPositionMutex()
    {
        return positionMutex;
    }
    
    XenomaiMutex &getLongPosMutex()
    {
        return longPosMutex;
    }
    
    XenomaiMutex &getCurrentHeightMutex()
    {
        return currentHeightMutex;
    }
	
protected:
	void run();
	unsigned long overruns;
	static const RTIME period = 1000000;
	bool runTask;
	bool taskFinished;
	bool loadingRoad;
	
	Road* currentRoad[12];
    double currentLongPos[12];
	
	osg::Vec3d roadPoint[12];
	
	std::vector<Road*> roadList[12];
	std::string currentRoadName[12];
	int currentRoadId[12];
	double currentHeight[12];
	bool singleRoadSwitch[12];
	bool leftRoadSwitch[12];
	bool roadListChanged[12];
	double roadHeightIncrement[12];
	double roadHeightDifference[12];
	double roadHeightDelta[12];
	double roadHeightIncrementDelta;
	double roadHeightIncrementMax;
	double maxHeight;
	double minHeight;
	double maxHeightGap;
};

inline void RoadPointFinder::setRoad(Road *&road, int contactPointNumber)
{
	
	currentRoad[contactPointNumber] = road;
	
}

inline Road* RoadPointFinder::getRoad(int contactPointNumber)
{
	
	return currentRoad[contactPointNumber];
	
}

inline void RoadPointFinder::setPoint(osg::Vec3d inVector, int contactPointNumber)
{
	
	roadPoint[contactPointNumber] = inVector;
	
}

inline double RoadPointFinder::getHeight(int contactPointNumber)
{
	return currentHeight[contactPointNumber];
}

inline void RoadPointFinder::setHeight(double height, int contactPointNumber)
{
	currentHeight[contactPointNumber] = height;
}

inline void RoadPointFinder::setLongPos(double inLongPos, int contactPointNumber)
{
	
	currentLongPos[contactPointNumber] = inLongPos;
	
}

inline double RoadPointFinder::getLongPos(int contactPointNumber)
{
	
	return currentLongPos[contactPointNumber];
	
}

inline void RoadPointFinder::checkLoadingRoads(bool loadingState)
{
	loadingRoad = loadingState;
}

#endif