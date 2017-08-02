#ifndef __RoadPointFinder_h
#define __RoadPointFinder_h

#include "XenomaiTask.h"
#include "CanOpenController.h"
#include "XenomaiSteeringWheel.h"

class RoadPointFinder : public XenomaiTask
{
public:
	RoadPointFinder();
	~RoadPointFinder();
	
	
	
protected:
	void run();
	unsigned long overruns;
	static const RTIME period = 1000000;
	bool runTask;
	bool taskFinished;
	
	
};

#endif