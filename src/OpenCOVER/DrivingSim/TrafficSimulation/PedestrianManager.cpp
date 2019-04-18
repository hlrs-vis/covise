/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PedestrianManager.h"
#include "TrafficSimulation.h"
#include <osg/Geometry>

/**
 * Create PedestrianManager as a singleton
 */
PedestrianManager *PedestrianManager::__instance = NULL;
PedestrianManager *PedestrianManager::Instance()
{
    if (__instance == NULL)
    {
        __instance = new PedestrianManager();
    }
    return __instance;
}
void PedestrianManager::Destroy()
{
    delete __instance;
    __instance = NULL;
}
PedestrianManager::PedestrianManager()
    : roadSystem(RoadSystem::Instance())
    , spawnRange(-1.0)
    , spawnRangeSq(1.0e5)
    , reportTimer(0.0)
    , repeatTime(-1.0)
    , nextTime(-1.0)
    , avoidCount(3)
    , avoidTime(3.0)
    , autoFiddleyards(false)
    , movingFiddleyards(false)
{
}

/**
 * Populate the road system with fiddleyards
 */
void PedestrianManager::generateAutoFiddleyards()
{
    autoFiddleyards = true;

    //TODO: Only active fiddleyards should be those that are farthest from the viewer; directions of source/sink are then important
    //      Activate all sinks _outside_ the viewer-range, and activate all sources _inside_ the viewer range
    //      Sources point towards viewer, sinks point away from viewer
    //      Instead of sinks, just remove any pedestrian outside of range during moveAllPedestrians()
    //      Source from slightly inside the "removal" radius; allow pedestrians to walk in both directions (e.g., "with" and "towards" viewer)
    //
    // Set sources at intervals on all roads with sidewalks (peds in both directions)
    // Source only active if within spawnRange and distSq is close to spawnRangeSq (e.g., within 4 meters)
    // Sink all pedestrians outside of spawnRange (during moveAll)
    int numRoads = roadSystem->getNumRoads();
    for (int i = 0; i < numRoads; i++)
    {
        // Consider each road in the road system
        Road *road = roadSystem->getRoad(i);
        std::string roadId = roadSystem->getRoadId(road);
        if (roadId.length() < 1)
            return;

        // Create a fiddleyard (for now)
        std::stringstream roadNum;
        roadNum << i;
        std::string fyId = std::string("r") + roadNum.str();
        PedFiddleyard *fy = new PedFiddleyard(fyId, fyId, roadId);

        // If this road yields any sources/sinks, we will add this fiddleyard to the roadSystem
        bool hasSrcSink = false;

        double length = road->getLength();
        for (double p = 0.0; p < length; p += 20.0)
        {
            // At each 20-meter interval on this road, find sidewalk(s) and place fiddleyards
            std::vector<int> laneNums = road->getSidewalks(p);
            std::vector<int>::iterator laneIt = laneNums.begin();
            for (; laneIt < laneNums.end(); laneIt++)
            {
                int lane = (*laneIt);

                // There is a sidewalk, so we'll make a source/sink; generate unique id for this position/laneNum
                std::stringstream pos;
                pos << p;
                std::stringstream ln;
                ln << lane;
                std::string posId = std::string("p") + pos.str() + std::string("l") + ln.str();

                // Determine fiddleyard parameters
                //TODO: parse defaults from somewhere? e.g., <pedFiddleyardDefaults .../> entry in .xodr
                //
                // For source
                std::string srcId = std::string(posId + "src");
                double starttime = 0.0;
                double repeattime = 15.0;
                double timedev = 8.0;
                int srcDir = 0;
                double vOffSrc = 0.0;
                double vel = 1.3;
                double velDev = 0.3;
                double acc = 0.65;
                double accDev = 0.15;
                /*
            //
            // For sink
            std::string sinkId = std::string(posId+"snk");
            int sinkDir = -1; //TODO: sink away from viewer position
            double vOffSink = 5.0;
            double sinkProb = 1.0;
*/
                // Create a source and sink
                PedSource *src = new PedSource(fy, srcId, starttime, repeattime, timedev, lane, srcDir, p, vOffSrc, vel, velDev, acc, accDev);
                //            PedSink* sink = new PedSink(fy, sinkId, sinkProb, lane, sinkDir, (p - 1.0 * srcDir), vOffSink);
                fy->addSource(src);
                //            fy->addSink(sink);
                hasSrcSink = true;

                // Add pedestrians to source
                //TODO: don't hardcode; one for default and one for each template?
                src->addPedRatio("", 1.0); // default, i.e., no templateId
                src->addPedRatio("kid", 1.0);
                src->addPedRatio("skeleton", 1.0);

                // Print status
                fprintf(stderr, "PedestrianManager::generateAutoFiddleyards(): Added source %s to fiddleyard %s on road %s\n", srcId.c_str(), fyId.c_str(), roadId.c_str());
            }
        }

        // If any sources/sinks were created, add it to the roadSystem
        if (hasSrcSink)
            roadSystem->addPedFiddleyard(fy);
    }
}

/**
 * Set the moving fiddleyard flag, and offset the initial position of the viewer to force an immediate update
 */
void PedestrianManager::enableMovingFiddleyards()
{
    movingFiddleyards = true;

    osg::Vec3d viewerPosWld = opencover::cover->getViewerMat().getTrans();
    osg::Vec3d viewPos = viewerPosWld * opencover::cover->getInvBaseMat();
    viewPosLastUpdated = osg::Vec3d(viewPos.x() + 1.0e10, viewPos.y() + 1.0e10, viewPos.z() + 1.0e10);
}

/**
 * Find the moving fiddleyards in the system:
 *   Sinks defined implicitly by removing all pedestrians outside of the spawn radius
 *   Sources defined expicitly on roads at positions near (but inside) the radius, ped directions both with & towards the viewer
 */
void PedestrianManager::generateMovingFiddleyards()
{
    // Check for distance the viewpoint has moved since the last clear, only re-generate fiddleyards when 2 meters have been moved
    double viewDistMovedSq = squaredDistTo(viewPosLastUpdated.x(), viewPosLastUpdated.y(), viewPosLastUpdated.z());
    if (viewDistMovedSq < 4.0)
        return;

    // Update viewer position
    osg::Vec3d viewerPosWld = opencover::cover->getViewerMat().getTrans();
    viewPosLastUpdated = viewerPosWld * opencover::cover->getInvBaseMat();

    // Remove all existing fiddleyards
    roadSystem->clearPedFiddleyards();

    // Initialize unique id counter
    int curSrcId = 0;

    // Find positions on roads just inside the radius
    int numRoads = roadSystem->getNumRoads();
    for (int i = 0; i < numRoads; i++)
    {
        // Consider each road in the road system
        Road *road = roadSystem->getRoad(i);
        std::string roadId = roadSystem->getRoadId(road);
        if (roadId.length() < 1)
            return;

        // Create a fiddleyard (for now)
        std::stringstream roadNum;
        roadNum << i;
        std::string fyId = std::string("r") + roadNum.str();
        PedFiddleyard *fy = new PedFiddleyard(fyId, fyId, roadId);
        bool addFyToSys = false;

        // Search the length of the road for (at most two)points inside the spawn radius
        double length = road->getLength();
        //
        // Search for first point (point inside radius closest to start of the road)
        bool searching = true;
        double searchPoint = 0.0;
        bool lastInRadius = false;
        double lastValidPoint = -1.0;
        bool pointFound = false;
        double jumpDist = length / 2;
        while (searching)
        {
            // Measure distance to search point
            double distSq = squaredDistTo(road->getCenterLinePoint(searchPoint).x(),
                                          road->getCenterLinePoint(searchPoint).y(),
                                          road->getCenterLinePoint(searchPoint).z());
            if (distSq <= spawnRangeSq)
            {
                // Within radius, check for a sidewalk at the point
                if (road->getSidewalks(searchPoint).size() > 0)
                {
                    // There is a sidewalk, this point is valid
                    lastInRadius = true;
                    lastValidPoint = searchPoint;
                    pointFound = true;
                    jumpDist = jumpDist / 2;
                }
            }

            // Calculate next search point
            if (lastInRadius)
                searchPoint -= jumpDist;
            else
                searchPoint += jumpDist;
            if (searchPoint < 0.0)
                searchPoint = 0.0;
            if (searchPoint > length)
                searchPoint = length;

            // Exit loop if jump distance becomes small enough or if we got close enough to the radius
            if (jumpDist < 4.0 || distSq < 16.0)
                searching = false;
        }
        // Check whether the first point was found
        if (pointFound)
        {
            // Print status
            fprintf(stderr, "PedestrianManager::generateMovingFiddleyards(): Point for near end of road %s at %.2f\n", roadId.c_str(), lastValidPoint);

            // Create sources (one in each direction and one on each sidewalk), add to fiddleyard, and add fiddleyard to roadSystem
            std::vector<int> laneNums = road->getSidewalks(lastValidPoint);
            std::vector<int>::iterator laneIt = laneNums.begin();
            for (; laneIt < laneNums.end(); laneIt++)
            {
                std::stringstream curSrcIdString1;
                curSrcIdString1 << curSrcId++;
                std::stringstream curSrcIdString2;
                curSrcIdString2 << curSrcId++;

                // Determine fiddleyard parameters
                //TODO: parse defaults from somewhere? e.g., <pedFiddleyardDefaults .../> entry in .xodr
                //
                // For source
                std::string srcId1 = std::string("src" + curSrcIdString1.str());
                std::string srcId2 = std::string("src" + curSrcIdString2.str());
                double starttime = 0.0;
                double repeattime = 15.0;
                double timedev = 0.0;
                int lane = (*laneIt);
                int dir = 1;
                double vOffSrc = 0.0; // Start on sidewalk
                double vel = 1.3;
                double velDev = 0.2;
                double acc = 0.65;
                double accDev = 0.1;

                // Create a source
                PedSource *src1 = new PedSource(fy, srcId1, starttime, repeattime, timedev, lane, dir, lastValidPoint, vOffSrc, vel, velDev, acc, accDev);
                PedSource *src2 = new PedSource(fy, srcId2, starttime + 2, repeattime, timedev, lane, -1 * dir, lastValidPoint + 0.1, vOffSrc, vel, velDev, acc, accDev);
                fy->addSource(src1);
                fy->addSource(src2);
                addFyToSys = true;

                // Add pedestrians to source
                //TODO: don't hardcode; one for default and one for each template?
                src1->addPedRatio("", 1.0); // default, i.e., no templateId
                src1->addPedRatio("kid", 1.0);
                src1->addPedRatio("skeleton", 1.0);
                src2->addPedRatio("", 1.0); // default, i.e., no templateId
                src2->addPedRatio("kid", 1.0);
                src2->addPedRatio("skeleton", 1.0);

                // Print status
                fprintf(stderr, "PedestrianManager::generateFiddleyards(): Added source1 %s and source2 %s to fiddleyard %s on road %s\n", srcId1.c_str(), srcId2.c_str(), fyId.c_str(), roadId.c_str());
            }
        }
        //
        // Search for second point (point inside radius closest to end of the road)
        //TODO: start at the road's length, and check backwards toward the first point
        //      only perform this search if a first point was found (otherwise the whole road will already have been checked)
    }
}

/**
 * Set the interval at which reports about the pedestrian system will be printed to stderr
 */
void PedestrianManager::setReportInterval(const double i)
{
    repeatTime = i;
    nextTime = i;
}

/**
 * Set the number of upcoming pedestrians that will be examined for avoidance
 */
void PedestrianManager::setAvoidCount(const int c)
{
    int count = c < 0 ? -c : c;
    avoidCount = count;
}

/**
 * Set the time-gap that will be used for judging avoidance
 */
void PedestrianManager::setAvoidTime(const double t)
{
    double time = t < 0 ? -t : t;
    avoidTime = time;
}

/**
 * Remove the given pedestrian before the next update
 */
void PedestrianManager::scheduleRemoval(Pedestrian *p)
{
    // Add to removal list
    pedestrianRemovalList.push_back(p);
}

/**
 * Add a new pedestrian to the manager's overall list
 */
void PedestrianManager::addPedestrian(Pedestrian *p)
{
    // Add to overall list
    pedestrianOverallList.push_back(p);
}

/**
 * Remove the given pedestrian from the manager's lists, and then delete
 */
void PedestrianManager::removePedestrian(Pedestrian *p, Road *r, const int l)
{
    // Remove from the manager's overall list and the road/lane list
    if (p != NULL)
    {
        // Overall list
        std::vector<Pedestrian *>::iterator pedIt = std::find(pedestrianOverallList.begin(), pedestrianOverallList.end(), p);
        if (pedIt < pedestrianOverallList.end())
        {
            pedestrianOverallList.erase(pedIt);
        }

        // Road/lane list
        if (r != NULL && p->isOnRoad())
        {
            removeFromRoad(p, r, l);
        }
    }

    // Report removal
    if (p != NULL && p->getDebugLvl() > 0)
        fprintf(stderr, " Pedestrian '%s': removed at %f; %d pedestrians left\n", p->getName(), p->getU(), numPeds());

    // Remove from the scenegraph
    if (p != NULL)
    {
        PedestrianFactory::Instance()->deletePedestrian(p);
    }
}

/**
 * Remove a pedestrian from a road/lane list
 */
void PedestrianManager::removeFromRoad(Pedestrian *p, Road *r, const int l)
{
    // Remove from the lane's list
    std::pair<Road *, int> pedRoad(r, l);
    std::vector<Pedestrian *>::iterator pedIt = std::find(pedestrianListMap[pedRoad].begin(), pedestrianListMap[pedRoad].end(), p);
    if (pedIt < pedestrianListMap[pedRoad].end())
    {
        pedestrianListMap[pedRoad].erase(pedIt);
        p->onRoad(false);
    }

    //NOTE: should be accompanied by a call to addToRoad() at some point, to ensure the pedestrian is still monitored by the manager (for avoidance/removal purposes)

    // Resolve any avoidance issues
    if (p->getAvoiding() != NULL)
    {
        p->setAvoiding(NULL, 0);
    }
    if (p->getPassing() != NULL)
    {
        p->getPassing()->setPassedBy(NULL, 0);
        p->setPassing(NULL, 0);
    }
    if (p->getWaitingFor() != NULL)
    {
        p->setWaitingFor(NULL, 0);
    }
    if (p->getPassedBy() != NULL)
    {
        p->getPassedBy()->setPassing(NULL, 0);
        p->setPassedBy(NULL, 0);
    }
}

/**
 * Add a pedestrian to a road/lane list
 */
void PedestrianManager::addToRoad(Pedestrian *p, Road *r, const int l)
{
    // Reset sinks
    p->passSink("");

    // Add to the lane's list
    std::pair<Road *, int> pedRoad(r, l);

    // Find the correct position (sorted by increasing location on road, p->getU() )
    std::vector<Pedestrian *>::iterator pedIt = pedestrianListMap[pedRoad].begin();
    for (; pedIt < pedestrianListMap[pedRoad].end(); pedIt++)
    {
        // If this is the position, insert pedestrian 'p' before this pedestrian
        if (p->getU() < (*pedIt)->getU())
            break;

        // If 'p' is the only pedestrian, or p->getU() is less than all other pedestrians, insert before the end of the list
    }
    pedestrianListMap[pedRoad].insert(pedIt, p);

    // Pedestrian is now on a road
    p->onRoad(true);

    //NOTE: should be accompanied by a call to removeFromRoad() at some point, to ensure the pedestrian is not present in two lists
}

/**
 * Remove a pedestrian from one road/lane list, and add to another road/lane list
 */
void PedestrianManager::changeRoad(Pedestrian *p, Road *fromR, const int fromL, Road *toR, const int toL)
{
    // Remove from fromRoad
    removeFromRoad(p, fromR, fromL);

    // Add to toRoad
    addToRoad(p, toR, toL);
}

/**
 * Return the number of pedestrian currently present in the scenegraph (i.e., number of pedestrians that have been created, but not deleted)
 */
int PedestrianManager::numPeds() const
{
    return pedestrianOverallList.size();
}

/**
 * For each pedestrian, perform avoidance as necessary and tell them to move based on the last timestep
 */
void PedestrianManager::moveAllPedestrians(const double dt)
{
    // Maintain a record of which lists contain pedestrians, to perform sorting at the end
    std::vector<std::vector<Pedestrian *> *> activeLists;

    // Every [repeatTime], report how many pedestrians are present and how many are active
    bool report = false;
    int onRoadPeds = 0;
    int activePeds = 0;
    int inactivePeds = 0;
    if (repeatTime >= 0)
    {
        reportTimer += dt;
        if (reportTimer >= nextTime)
        {
            report = true;
            nextTime += repeatTime;
        }
    }

    // Perform avoidance for pedestrians that are on a road and active, and call each pedestrian's move function
    std::vector<Pedestrian *>::iterator pedIt = pedestrianOverallList.begin();
    for (; pedIt < pedestrianOverallList.end(); pedIt++)
    {
        // Check to see if pedestrian is active
        Pedestrian *p = (*pedIt);
        if (p != NULL)
        {
            p->updateActive();

            // Collect report metrics
            if (report)
            {
                if (p->isOnRoad())
                    onRoadPeds++;
                if (p->isActive())
                    activePeds++;
                else
                    inactivePeds++;
            }

            // Perform avoidance if this pedestrian is on a road and is active
            if (p->isOnRoad() && p->isActive())
            {
                performAvoidance(p, avoidTime);
            }

            // Add his road/lane list to the active lists
            if (p->isOnRoad())
            {
                // Add to list of active lists
                std::pair<Road *, int> pedRoad(p->getRoad(), p->getLaneNum());

                // Ensure this list is added exactly once
                std::vector<std::vector<Pedestrian *> *>::iterator searchIt = std::find(activeLists.begin(), activeLists.end(), &pedestrianListMap[pedRoad]);
                if (searchIt >= activeLists.end())
                {
                    activeLists.push_back(&pedestrianListMap[pedRoad]);
                }
            }

            // Move the pedestrian
            p->move(dt);
			if (p->isActive())
				p->getPedestrianGeometry()->update(dt);

            // auto-sink: Remove pedestrians outside the viewer's range (new ones should be added by auto-sources)
            if (autoFiddleyards || movingFiddleyards)
            {
                if (!p->isWithinRange(spawnRange + 10))
                    scheduleRemoval(p); //TODO: add 10m to account for distance between center of road and sidewalk
            }
        }
        else
        {
            // Pointer is NULL and should not be in overall list
            fprintf(stderr, "PedestrianManager::moveAllPedestrians(): NULL pointer in overall list\n");
        }
    }

    // Sort all lists that are active
    std::vector<std::vector<Pedestrian *> *>::iterator listIt = activeLists.begin();
    for (; listIt < activeLists.end(); listIt++)
    {
        std::sort((*listIt)->begin(), (*listIt)->end(), Pedestrian::comparePedestrians);
    }

    // Report metrics
    if (report)
    {
        // Print report
        fprintf(stderr, "PedestrianManager::moveAllPedestrians(): peds: %d, onRoad: %d, active: %d, inactive: %d\n", numPeds(), onRoadPeds, activePeds, inactivePeds);
    }
}

/**
 * Check all pedestrian fiddleyards, to:
 *   spawn any new pedestrians from sources that have reached their repeatTime (and are within spawnRange), and
 *   remove any pedestrians that have reached a sink
 */
void PedestrianManager::updateFiddleyards(const double dt)
{
    // Perform any pending removals
    performRemovals();

    // Update the positions of "moving" fiddleyards
    if (movingFiddleyards)
        generateMovingFiddleyards();

    if (roadSystem)
    {
        // Examine sources/sinks of each fiddleyard in the roadsystem
        for (int i = 0; i < roadSystem->getNumPedFiddleyards(); i++)
        {
            PedFiddleyard *fy = roadSystem->getPedFiddleyard(i);
            double time = fy->incrementTimer(dt);
            std::string roadId = fy->getRoadId();

            // Remove any pedestrians that reached a sink
            const std::map<std::pair<int, double>, PedSink *> &sinkMap = fy->getSinkMap();
            for (std::map<std::pair<int, double>, PedSink *>::const_iterator sinkIt = sinkMap.begin(); sinkIt != sinkMap.end(); sinkIt++)
            {
                Road *road = roadSystem->getRoad(roadId);
                PedSink *sink = sinkIt->second;
                std::string sinkId = sink->getId();
                double prob = sink->getProbability();
                int lane = sink->getLane();
                int dir = sink->getDir();
                std::pair<Road *, int> pedRoad(road, lane);
                double sOff = sink->getSOffset();
                double vOff = sink->getVOffset();
                // vOffset is always away from the center lane
                if (lane > 0 && vOff < 0)
                    vOff = -vOff;
                if (lane < 0 && vOff > 0)
                    vOff = -vOff;

                // Queue pedestrians that have expired and need to be removed
                // (but don't remove them until the end, when iterations are complete)
                std::vector<Pedestrian *> removals;

                // Look forward to find those pedestrians "in" the sink
                //   i.e., any peds with u-value greater than sOff and a matching direction are candidates for removal,
                //         as long as they have not been previously allowed to continue due to the sink's probability
                std::vector<Pedestrian *>::iterator pedIt = pedestrianListMap[pedRoad].begin();
                for (; pedIt < pedestrianListMap[pedRoad].end(); pedIt++)
                {
                    Pedestrian *ped = (*pedIt);

                    if (ped->getU() >= sOff - 0.5 && ped->getU() <= sOff + 0.5 && (ped->getDir() == dir || dir == 0))
                    {
                        if (!ped->isLeavingSidewalk() && ped->passSink().compare(sinkId) != 0)
                        {
                            // Ped has _not_ yet passed this sink and continued, so possibly remove
                            double rand = TrafficSimulation::instance()->getZeroOneRandomNumber();
                            if (rand <= prob)
                            {
                                // Pedestrian will be removed here
                                ped->leaveSidewalk(vOff);
                            }
                            else
                            {
                                // Pedestrian will not be removed in this sink
                                ped->passSink(sinkId);
                            }
                        }
                    }
                }
            }

            // Spawn new pedestrians if any sources have reached their repeatTime
            const std::map<std::pair<int, double>, PedSource *> &sourceMap = fy->getSourceMap();
            for (std::map<std::pair<int, double>, PedSource *>::const_iterator sourceIt = sourceMap.begin(); sourceIt != sourceMap.end(); sourceIt++)
            {
                PedSource *source = sourceIt->second;

                // Only perform spawns if the maximum number of pedestrians hasn't been exceeded
                if (PedestrianFactory::Instance()->maxPeds() < 0 || numPeds() < PedestrianFactory::Instance()->maxPeds())
                {
                    // Check whether this source has reached its spawn time
                    if (source->spawnNextPed(time))
                    {
                        // If spawnRange is limited, check distance from source to viewer
                        double distSq = -1.0e5;
                        if (spawnRange > 0.0)
                        {
                            distSq = squaredDistTo(roadSystem->getRoad(roadId)->getCenterLinePoint(source->getSOffset()).x(),
                                                   roadSystem->getRoad(roadId)->getCenterLinePoint(source->getSOffset()).y(),
                                                   roadSystem->getRoad(roadId)->getCenterLinePoint(source->getSOffset()).z());
                        }

                        // Check conditions (depends on auto-, moving-, or normal-fiddleyards)
                        bool spawn = false;
                        if (autoFiddleyards)
                        {
                            if (distSq <= spawnRangeSq)
                            {
                                // Within spawn range
                                if (spawnRange - sqrt(distSq) <= 8)
                                {
                                    // Within 8 meters of spawn range
                                    spawn = true;
                                }
                            }
                        }
                        else if (distSq <= spawnRangeSq)
                            spawn = true;
                        // Only spawn pedestrians if this fiddleyard is within the viewer's range (or if no range was set)
                        if (spawn)
                        {
                            // Update source's timer
                            source->updateTimer(time, TrafficSimulation::instance()->getZeroOneRandomNumber());

                            // Choose a pedestrian to spawn (based on ratio = numerator/total)
                            double targetRatio = TrafficSimulation::instance()->getZeroOneRandomNumber() * source->getOverallRatio();
                            std::string tmplString = source->getPedByRatio(targetRatio);

                            // Generate pedestrian's name
                            std::ostringstream timeStringStream;
                            timeStringStream << time;
                            std::string nameString = std::string("fy") + fy->getId() + std::string("_src") + source->getId() + std::string("_") + ((tmplString.length() > 0) ? tmplString : std::string("default")) + std::string("_") + timeStringStream.str();

                            // Determine starting position
                            int lane = source->getLane();
                            int dir = source->getDir();
                            if (dir == 0)
                                dir = (TrafficSimulation::instance()->getZeroOneRandomNumber() <= 0.5) ? -1 : 1;
                            double sOff = source->getSOffset();
                            double vOff = source->getVOffset();
                            // vOffset is always away from the center lane
                            if (lane > 0 && vOff < 0)
                                vOff = -vOff;
                            if (lane < 0 && vOff > 0)
                                vOff = -vOff;

                            // Determine velocity
                            double vel = 0;
                            vel = source->getStartVelocity() - source->getStartVelocityDeviation() + 2 * source->getStartVelocityDeviation() * TrafficSimulation::instance()->getZeroOneRandomNumber();
                            double acc = 0;
                            acc = source->getAccel() - source->getAccelDev() + 2 * source->getAccelDev() * TrafficSimulation::instance()->getZeroOneRandomNumber();

                            // Create the new pedestrian
                            PedestrianFactory::Instance()->createPedestrian(nameString, tmplString, roadId, lane, dir, sOff, vOff, vel, acc);
                        }
                    }
                }
            }
        }
    }
}

/**
 * Perform all scheduled removals
 */
void PedestrianManager::performRemovals()
{
    // Remove each pedestrian in this list
    for (int i = 0; i < pedestrianRemovalList.size(); i++)
    {
        removePedestrian(pedestrianRemovalList[i], pedestrianRemovalList[i]->getRoad(), pedestrianRemovalList[i]->getLaneNum());
    }

    // Clear this list
    pedestrianRemovalList.clear();
}

/**
 * Returns a list containing the (up to) n closest neighbors of the given pedestrian in his direction of travel
 */
std::vector<Pedestrian *> PedestrianManager::getClosestNeighbors(Pedestrian *p, int n)
{
    // Results will be stored in a vector
    std::vector<Pedestrian *> result;

    // Find the position of the given pedestrian in the sorted road/lane list
    int pDir = p->getDir();
    std::pair<Road *, int> pedRoad(p->getRoad(), p->getLaneNum());
    std::vector<Pedestrian *>::iterator pedIt = pedestrianListMap[pedRoad].begin();
    int i = 0;
    while ((*pedIt) != p)
    {
        pedIt++;
        i++;
    }

    // Store (up to) the next n pedestrians in the pedestrian's direction of travel
    for (int j = 0; j < n; j++)
    {
        if ((pDir < 0) ? (i > j) : (pedestrianListMap[pedRoad].size() - 1 - i > j))
        {
            if (pDir < 0)
                pedIt--;
            else
                pedIt++;
            result.push_back(*pedIt);
        }
        else
            break;
    }

    // Return the resulting list
    return result;
}

/**
 * Compare the pedestrian's position to all other pedestrians on the road
 * Avoid any pedestrians that will have caused a collision by the given time (either by passing, waiting, or avoiding)
 */
void PedestrianManager::performAvoidance(Pedestrian *p, const double time)
{
    // Check for possible avoidance/passing situations
    //   1) Avoiding collision -> both parties bear to their right
    //   2) Passing a slower pedestrian -> pass on the right, other party remains in center
    //   3) Being passed by a faster ped -> remain in center, allow passing on the right
    //NOTE: "to the right" means a positive vTarget on the lane, not necessarily to the pedestrian's right

    // Check/update existing conditions
    if (p->getAvoiding() != NULL)
    {
        if ((p->getDir() > 0) ? (p->getU() > p->getAvoiding()->getU()) : (p->getU() < p->getAvoiding()->getU()))
        {
            // Have avoided the target
            p->setAvoiding(NULL, 0);
        }
    }
    if (p->getPassing() != NULL)
    {
        if ((p->getDir() > 0) ? (p->getU() > p->getPassing()->getU()) : (p->getU() < p->getPassing()->getU()))
        {
            // Have passed the target
            p->setPassing(NULL, 0);
        }
    }
    if (p->getWaitingFor() != NULL)
    {
        if (p->getWaitingFor()->getAvoiding() == NULL && p->getWaitingFor()->getPassing() == NULL && p->getWaitingFor()->getWaitingFor() == NULL)
        {
            // Target is no longer avoiding/passing/waiting, no longer need to wait
            p->setWaitingFor(NULL, 0);
        }
    }
    if (p->getPassedBy() != NULL)
    {
        if ((p->getDir() > 0) ? (p->getU() < p->getPassedBy()->getU()) : (p->getU() > p->getPassedBy()->getU()))
        {
            // Has been passed by the other party, no longer being passed
            p->setPassedBy(NULL, 0);
        }
    }

    // If all conditions have been cleared, check for new conditions
    if (p->getAvoiding() == NULL && p->getPassing() == NULL && p->getWaitingFor() == NULL && p->getPassedBy() == NULL)
    {
        // Compare position to closest neighbors on this road/lane
        std::vector<Pedestrian *> closestPeds = getClosestNeighbors(p, avoidCount);
        std::vector<Pedestrian *>::iterator pedIt = closestPeds.begin();
        for (; pedIt < closestPeds.end(); pedIt++)
        {
            Pedestrian *i = (*pedIt);
            // Perform avoidance test between 'p' and 'i', if: they are different, both active, and both on the road
            if (p != i && i->isActive() && i->isOnRoad())
            {
                double collisionTime = collisionIn(p, i);
                if (collisionTime <= time)
                {
                    // Must either (a) avoid an opposite-direction pedestrian, (b) pass a slower pedestrian from behind, or (c) wait for a slower pedestrian to finish a pass

                    if (p->getDir() == i->getDir())
                    {
                        // Going in same direction, so either pass or wait
                        if (i->getAvoiding() != NULL || i->getPassing() != NULL || i->getWaitingFor() != NULL)
                        {
                            // i is avoiding or passing or waiting, so it isn't safe to pass
                            p->setWaitingFor(i, collisionTime);
                        }
                        else
                        {
                            // i is neither avoiding nor passing nor waiting, so pass
                            p->setPassing(i, collisionTime);
                            i->setPassedBy(p, collisionTime);
                        }
                    }
                    else
                    {
                        // Going in opposite direction, so either avoid or wait
                        if (p->getWaitingFor() != NULL || p->getPassedBy() != NULL)
                        {
                            // Waiting for someone ahead or being passed by someone from behind, so it isn't safe to avoid
                        }
                        else
                        {
                            // No obstacles, so avoid
                            p->setAvoiding(i, collisionTime);
                            p->executeWave();
                        }
                    }
                }
            }
        }
    }
}

/**
 * Return the amount of time until a collision between the two pedestrians, if a collision will have taken place within the given time limit
 */
double PedestrianManager::collisionIn(Pedestrian *a, Pedestrian *b) const
{
    // Determine search area for pedestrian 'a'
    int aDir = a->getDir();
    double aPos = a->getU();
    double aVel = a->getDuMax();

    // Calculate where pedestrian 'b' will be at the given time
    int bDir = b->getDir();
    double bPos = b->getU();
    double bVel = (aDir == bDir) ? b->getDu() : b->getDuMax(); // If direction is the same, use the smaller velocity

    // Calculate the time to collision
    double relVel = aVel * aDir - bVel * bDir;
    double collTime = (bPos - aPos) / relVel;

    // If the collision has not yet occurred, return the time (otherwise, return a large number)
    if (collTime < 0.0)
        return 1.0e8;
    else
        return collTime;
}

/**
 * Returns squared distance between the viewer position and the given point
 */
double PedestrianManager::squaredDistTo(double x, double y, double z)
{
    // Find viewer's position in object coordinates
    osg::Vec3d viewerPosWld = opencover::cover->getViewerMat().getTrans();
    osg::Vec3d viewPos = viewerPosWld * opencover::cover->getInvBaseMat();

    // Calculate distance squared to viewer's position
    double xDiffSq = (x - viewPos.x()) * (x - viewPos.x());
    double yDiffSq = (y - viewPos.y()) * (y - viewPos.y());
    double zDiffSq = (z - viewPos.z()) * (z - viewPos.z());
    return xDiffSq + yDiffSq + zDiffSq;
}
