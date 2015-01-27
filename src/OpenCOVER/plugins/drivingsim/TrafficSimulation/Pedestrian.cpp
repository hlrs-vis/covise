/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Pedestrian.h"
#include "TrafficSimulationPlugin.h"

/**
 * Create a new pedestrian
 */
Pedestrian::Pedestrian(std::string &n, Road *startRoad, int startLaneNum, int startDir, double startPos, double startVOff, double startVel, double startAcc, double headingAdj, int dbg, PedestrianGeometry *g)
    : debugLvl(dbg)
    , name(n)
    , geometry(g)
    , timer(0.0)
    , pedParams(headingAdj)
    , pedLoc(startRoad, startLaneNum, NULL, NULL, NULL, startDir)
    , pedState(true)
    , avoiding(NULL)
    , passing(NULL)
    , waitingFor(NULL)
    , passedBy(NULL)
{
    setDvMax(startVel / 10); // normal lateral velocity
    setDuMax(startVel, startAcc); // normal traveling velocity

    if (startRoad != NULL && startPos >= 0.0)
    {
        // Starting road/direction
        pedLoc.road = startRoad;
        pedLoc.dir = startDir;

        // Starting position
        pedState.u = startPos;

        // Find center of lane
        pedLoc.laneSec = startRoad->getLaneSection(startPos);
        Vector2D laneCenter = pedLoc.laneSec->getLaneCenter(startLaneNum, startPos);
        pedState.v = laneCenter[0];

        // Determine initial vOffset
        if (fabs(startVOff) > getLaneWidth(pedLoc.laneSec, startLaneNum, startPos) / 2)
        {
            // Place pedestrian at given vOffset, and find a sidewalk by decreasing fabs(startVOff) to 0
            pedState.searchSidewalk = true;
            pedState.foundSidewalk = false;
            pedState.vOff = startVOff;
            pedState.vStart = startVOff;
            pedState.vTgt = 0.0;
            pedState.v += startVOff;
            // Add to road once the sidewalk is reached
        }
        else
        {
            PedestrianManager::Instance()->addToRoad(this, startRoad, startLaneNum);
        }

        // Get starting lane
        pedLoc.laneNum = startLaneNum;
        pedLoc.lane = pedLoc.laneSec->getLane(startLaneNum);

        // Set starting crosswalk
        pedLoc.crosswalk = NULL;

        // Find starting heading
        pedState.hdg = M_PI / 2 * (startDir);
    }
    else
        printDebug(0, "%s is invalid", ((startPos < 0) ? "position" : "road (NULL)"));

    if (geometry != NULL)
    {
        geometry->setPedestrian(this);

        // Update geometry
        updateGeometry();
    }
    else
        printDebug(0, "geometry (NULL) is invalid");
}
Pedestrian::~Pedestrian()
{
    // Clear crosswalk conflicts
    if (pedLoc.crosswalk)
    {
        pedLoc.crosswalk->exitPedestrian(pedState.crossId, opencover::cover->frameTime());
        pedLoc.crosswalk->pedestrianDoneWaiting(pedState.crossWaitId, opencover::cover->frameTime());
    }

    // Clear passing conflicts
    if (passing != NULL)
    {
        passing->setPassedBy(NULL, 0);
    }
    if (passedBy != NULL)
    {
        passedBy->setPassing(NULL, 0);
    }

    // Clear avoidance conflicts
    setAvoiding(NULL, 0);
    setPassing(NULL, 0);
    setWaitingFor(NULL, 0);
    setPassedBy(NULL, 0);

    // Delete geometry
    delete geometry;
}

/**
 * Determine whether pedestrian is active or not, adjust settings appropriately
 */
void Pedestrian::updateActive()
{
    setActive(geometry->isGeometryWithinLOD());
}

/**
 * Determine whether pedestrian is within the given range
 */
bool Pedestrian::isWithinRange(const double r) const
{
    return geometry->isGeometryWithinRange(r);
}

/**
 * Perform a one-time action, where idx is its position in the Cal3d .cfg file's animation list (first item has idx 0)
 */
void Pedestrian::executeWave()
{
    geometry->executeWave(0.9 + 0.3 * TrafficSimulationPlugin::plugin->getZeroOneRandomNumber());
}

/**
 * Perform a one-time action, where idx is its position in the Cal3d .cfg file's animation list (first item has idx 0)
 */
void Pedestrian::executeAction(const int idx)
{
    geometry->executeAction(idx, 0.9 + 0.3 * TrafficSimulationPlugin::plugin->getZeroOneRandomNumber());
}

/**
 * Set an active pedestrian inactive, or an inactive pedestrian active
 *  Active: full update of position in u- and v-direction, acceleration, avoidance, etc.
 *  Inactive: continues to move and navigate roadsystem with no avoidance checks, acceleration, etc.
 */
bool Pedestrian::setActive(bool act)
{
    // Prevent redundant changes
    if (pedState.active && !act)
    {
        // Active, going inactive
        pedState.active = false;
        setVTarget(pedState.vOff);

        //DEBUG
        if (debugLvl > 0)
            printDebug(2, "is now inactive");

        return true;
    }
    else if (!pedState.active && act)
    {
        // Inactive, going active
        pedState.active = true;

        //DEBUG
        if (debugLvl > 0)
            printDebug(2, "is now active");

        return true;
    }
    return false;
}

/**
 * Returns the width of the lane at the given position
 */
double Pedestrian::getLaneWidth(LaneSection *ls, const int ln, const double s) const
{
    double pos = s < 0 ? -s : s;
    // Get width at this position
    double width = ls->getLaneWidth(pos, ln);
    return fabs(width);
}
/**
 * Returns the width of the lane at the current position
 */
double Pedestrian::getWalkwayWidth() const
{
    return getLaneWidth(pedLoc.laneSec, pedLoc.laneNum, pedState.u);
}

/**
 * Notifies the pedestrian to leave the sidewalk and walk to the given vOffset, where he should then be removed
 */
void Pedestrian::leaveSidewalk(const double v)
{
    setVTarget(v);
    pedState.leavingSidewalk = true;
    pedState.leavingDone = false;

    //DEBUG
    if (debugLvl > 0)
        printDebug(3, "leaving the sidewalk, setting vTgt = %.2f m", v);
}

/**
 * Sets a flag as to whether the pedestrian is currently on a road
 */
void Pedestrian::onRoad(const bool r)
{
    pedState.onRoad = r;
    if (debugLvl > 0)
        printDebug(3, "now %s the road, lane# %d (dir %d)", (r ? "on" : "off"), pedLoc.laneNum, pedLoc.dir);
}

/**
 * Sets the maximum forward velocity and acceleration
 */
void Pedestrian::setDuMax(const double du, const double ddu)
{
    // Should be positive
    double vel = fabs(du);
    double accel = fabs(ddu);
    pedParams.duMax = vel;
    pedParams.dduMax = accel;

    pedState.duStart = pedState.du;
    pedState.duTgt = du;

    // Update animation based on current speed
    geometry->setWalkingSpeed(pedState.du);

    //DEBUG
    if (debugLvl > 0)
        printDebug(5, "max forward vel set to %.2f m/s; accel set to %.2f m/s", du, ddu);
}

/**
 * Sets the maximum forward velocity when waiting to pass a slower pedestrian
 */
void Pedestrian::setDuWait(const double du)
{
    // Should be positive
    double vel = fabs(du);
    pedParams.duWait = vel;
    pedParams.dduWait = vel;

    // Update animation based on current speed
    geometry->setWalkingSpeed(pedState.du);

    //DEBUG
    if (debugLvl > 0)
        printDebug(5, "waiting vel&accel set to %.2f m/s", du);
}

/**
 * Sets a lateral target for non-linear forward motion
 */
void Pedestrian::setVTarget(const double vTarget)
{
    if (!isLeavingSidewalk())
    {
        // Reset vStart to current offset
        pedState.vStart = pedState.vOff;
        pedState.vAccelDist = 0.0;

        // Reset transversal velocity to 0
        pedState.dv = 0.0;

        // Set target
        pedState.vTgt = vTarget;

        //DEBUG
        if (debugLvl > 0)
            printDebug(6, "lateral velocity reset to 0.0, lateral target set to %.2f", vTarget);
    }
}

/**
 * Sets the maximum lateral velocity
 */
void Pedestrian::setDvMax(const double dv)
{
    // Should be positive
    double vel = fabs(dv);
    pedParams.dvMax = vel;
    pedParams.ddvMax = vel;

    //DEBUG
    if (debugLvl > 0)
        printDebug(5, "max lateral vel&accel set to %.2f m/s", dv);
}

/**
 * Sets the lateral velocity when passing a slower pedestrian (must move to the side of the pedestrian before the collision occurs)
 */
void Pedestrian::setDvPass(const double dv)
{
    // Should be positive
    double vel = fabs(dv);
    pedParams.dvPass = vel;
    pedParams.ddvPass = vel;

    //DEBUG
    if (debugLvl > 0)
        printDebug(5, "passing lateral vel&accel set to %.2f m/s", dv);
}

/**
 * Avoid a pedestrian going in the opposing direction within the given time
 */
void Pedestrian::setAvoiding(Pedestrian *p, const double time)
{
    //DEBUG
    if (debugLvl > 0)
    {
        if (p != NULL)
            printDebug(4, "now avoiding %s, collision in %.2f", p->getName(), time);
        else
            printDebug(4, "done avoiding %s", (avoiding == NULL) ? "NULL" : avoiding->getName());
    }

    avoiding = p;

    if (p != NULL)
    {
        // Pedestrian should set a transversal offset target outside of the normal walking lane
        // Pedestrian should get there before given time
        double tgt = (getDir() > 0) ? (getWalkwayWidth() * 0.5) * 0.5 : -(getWalkwayWidth() * 0.5) * 0.5;
        double vel = (getDir() > 0) ? (getVOffset() - tgt) / (time / 2) : (tgt - getVOffset()) / (time / 2);

        // Don't allow the avoidance velocity to get too high
        vel = (fabs(vel) > 3.0) ? 3.0 : fabs(vel);

        setDvPass(vel);
        setVTarget(tgt);
    }
    else
    {
        // Just finished an avoidance
        setVTarget(0.0);
    }
}

/**
 * Pass a slower pedestrian within the given time
 */
void Pedestrian::setPassing(Pedestrian *p, const double time)
{
    //DEBUG
    if (debugLvl > 0)
    {
        if (p != NULL)
            printDebug(4, "now passing %s, collision in %.2f", p->getName(), time);
        else
            printDebug(4, "done passing %s", (passing == NULL) ? "NULL" : passing->getName());
    }

    passing = p;

    if (p != NULL)
    {
        // Pedestrian should set a transversal offset target outside of the normal walking lane
        // Pedestrian should get there before given time
        double tgt = (getDir() > 0) ? (getWalkwayWidth() * 0.5) * 0.5 : -(getWalkwayWidth() * 0.5) * 0.5;
        double vel = (getDir() > 0) ? (getVOffset() - tgt) / (time / 2) : (tgt - getVOffset()) / (time / 2);

        // Don't allow the avoidance velocity to get too high
        vel = (fabs(vel) > 3.0) ? 3.0 : fabs(vel);

        setDvPass(vel);
        setVTarget(tgt);
    }
    else
    {
        // Just finished a pass
        setVTarget(0.0);
    }
}

/**
 * Wait for a slower pedestrian
 */
void Pedestrian::setWaitingFor(Pedestrian *p, const double time)
{
    //DEBUG
    if (debugLvl > 0)
    {
        if (p != NULL)
            printDebug(4, "now waiting for %s, collision in %.2f", p->getName(), time);
        else
            printDebug(4, "done waiting for %s", (waitingFor == NULL) ? "NULL" : waitingFor->getName());
    }

    waitingFor = p;

    if (p != NULL)
    {
        setDuWait(p->getDuMax());
    }
    else
    {
        // Done waiting, so get back up to the target velocity
        pedState.duStart = pedState.du;
        pedState.duTgt = pedParams.duMax;
    }
}

/**
 * Be aware of being passed by a faster pedestrian
 */
void Pedestrian::setPassedBy(Pedestrian *p, const double time)
{
    //DEBUG
    if (debugLvl > 0)
    {
        if (p != NULL)
            printDebug(4, "now being passed by %s, collision in %.2f", p->getName(), time);
        else
            printDebug(4, "done being passed by %s", (passedBy == NULL) ? "NULL" : passedBy->getName());
    }

    passedBy = p;

    if (p != NULL)
    {
        double vel = (getDir() > 0) ? (getVOffset() / (time / 2)) : (-getVOffset() / (time / 2));

        // Don't allow the avoidance velocity to get too high
        vel = (fabs(vel) > 3.0) ? 3.0 : fabs(vel);

        setDvPass(vel);
        setVTarget(0.0);
    }
}

/**
 * Trigger a change in the pedestrian's position
 */
void Pedestrian::move(const double dt)
{
    // Update timer
    timer += dt;

    // Move the pedestrian according to normal rules (unless crossing the street or searching for sidewalk)
    if (!pedState.cross && !pedState.searchSidewalk && !pedState.leavingSidewalk)
    {
        // Set new longitudinal position
        setNewU(dt);

        // Determine whether this dt/movement has left the current road or reached a new lane section
        // If so, in either case, test if it is walkable (otherwise, turn around)
        checkForNewRoad();
        checkForNewLaneSection(dt);
        checkForNewCrosswalk();

        // Set new lateral position
        setNewV(dt);
    }
    else if (pedState.cross)
    {
        // Move the pedestrian through the crosswalk
        crossStreet(dt);
    }
    else if (pedState.searchSidewalk)
    {
        // Decrease fabs(vOffset) until sidewalk is found
        findSidewalk(dt);
    }
    else if (pedState.leavingSidewalk)
    {
        // Increase vOffset until vTgt is reached
        leavingSidewalk(dt);
    }

    if (pedState.active)
    {
        // Set heading, to face forward in the walking direction
        setNewHdg();
    }

    // Update pedestrian's geometry
    updateGeometry();
}

/**
 * Move in the forward (u-)direction
 */
void Pedestrian::setNewU(const double dt)
{
    if (pedState.active)
    {
        // Set acceleration
        pedState.ddu = (getWaitingFor() != NULL) ? (pedParams.dduWait) : ((pedState.duTgt < pedParams.duMax) ? 0.01 : pedParams.dduMax);

        // Set velocity based on currect acceleration
        if (pedState.du != ((getWaitingFor() != NULL) ? pedParams.duWait : pedState.duTgt))
        {
            // Need to accelerate
            if (pedState.du < ((getWaitingFor() != NULL) ? pedParams.duWait : pedState.duTgt))
            {
                pedState.du += pedState.ddu * dt;
                if (pedState.du > ((getWaitingFor() != NULL) ? pedParams.duWait : pedState.duTgt))
                    pedState.du = ((getWaitingFor() != NULL) ? pedParams.duWait : pedState.duTgt);
            }

            // Need to decelerate
            else if (pedState.du > ((getWaitingFor() != NULL) ? pedParams.duWait : pedState.duTgt))
            {
                pedState.du -= pedState.ddu * dt;
                if (pedState.du < ((getWaitingFor() != NULL) ? pedParams.duWait : pedState.duTgt))
                    pedState.du = ((getWaitingFor() != NULL) ? pedParams.duWait : pedState.duTgt);
            }

            // Update animation based on current speed
            geometry->setWalkingSpeed(pedState.du);
        }
        else
        {
            if (getWaitingFor() == NULL)
            {
                // Set a new velocity target up to 1/8 less than nominal velocity
                double velDiff = TrafficSimulationPlugin::plugin->getZeroOneRandomNumber() * (pedParams.duMax) / 8;
                pedState.duStart = pedState.du;
                pedState.duTgt = pedParams.duMax - velDiff;

                //DEBUG
                if (debugLvl > 0)
                    printDebug(6, "reached velocity target of %.2f m/s; new target %.2f m/s", pedState.du, pedState.duTgt);
            }
        }

        // Set position in the walking direction, and update total distance traveled
        pedState.u += getDir() * pedState.du * dt;
        pedState.s += pedState.du * dt;
    }
    else
    {
        // Pedestrian is inactive, just update position
        pedState.u += getDir() * pedParams.duMax * dt;
        pedState.s += pedParams.duMax * dt;
    }
}

/**
 * Check for entry into a new road due to last forward movement
 */
void Pedestrian::checkForNewRoad()
{
    if ((pedState.u > getRoad()->getLength()) || (pedState.u < 0.0))
    {
        // Pedestrian has left the current road

        // We may find a new road, or reverse direction on the current one
        Road *newRoad = getRoad();
        int newDir = -1 * getDir();
        double newPosition = pedState.u;
        Junction *newJunction;

        // Find the next connected road/junction, if there is one
        TarmacConnection *tarConn = NULL;

        // Get successor or predecessor, depending on direction
        tarConn = (getDir() > 0) ? getRoad()->getSuccessorConnection() : getRoad()->getPredecessorConnection();

        // If there is a connection, this might be the new road
        if (tarConn)
        {
            // There is a road
            if ((newRoad = dynamic_cast<Road *>(tarConn->getConnectingTarmac())))
            {
                newDir = tarConn->getConnectingTarmacDirection();
                newJunction = NULL;

                //DEBUG
                if (debugLvl > 0)
                    printDebug(3, "left road (%s), new road (%s) found", getRoad()->getName().c_str(), newRoad->getName().c_str());
            }

            // There is a junction
            else if ((newJunction = dynamic_cast<Junction *>(tarConn->getConnectingTarmac())))
            {
                PathConnectionSet tarConnSet = newJunction->getPathConnectionSet(getRoad());

                // Find the connecting sidewalk lane (if it exists)
                Road *roadWithSidewalk = NULL;
                int dirOfPath = 0;
                for (PathConnectionSet::iterator pcsIt = tarConnSet.begin(); pcsIt != tarConnSet.end(); pcsIt++)
                {
                    // Get the connecting road of the current pathConnection
                    PathConnection *pc = (*pcsIt);
                    Road *r = pc->getConnectingPath();

                    // Will need to get lane info
                    LaneSection *ls;
                    int ln;

                    // Check whether this is a path with a connected sidewalk
                    ls = r->getLaneSection(0.0);
                    ln = pc->getConnectingLane(pedLoc.laneNum);
                    if (ls != NULL && ls->isSidewalk(ln))
                    {
                        //NOTE: If road directions match, lane signs must match
                        if ((pc->getConnectingPathDirection() == getDir()) ? ((pedLoc.laneNum < 0 && ln < 0) || (pedLoc.laneNum > 0 && ln > 0)) : ((pedLoc.laneNum < 0 && ln > 0) || (pedLoc.laneNum > 0 && ln < 0)))
                        {
                            roadWithSidewalk = r;
                            dirOfPath = pc->getConnectingPathDirection();
                            break;
                        }
                    }
                }

                // Check whether a sidewalk was found
                if (roadWithSidewalk != NULL && dirOfPath != 0)
                {
                    newRoad = roadWithSidewalk;
                    newDir = dirOfPath;

                    //DEBUG
                    if (debugLvl > 0)
                        printDebug(3, "left road (%s), new junction found with sidewalk on road (%s)", getRoad()->getName().c_str(), newRoad->getName().c_str());
                }
                else
                {
                    //DEBUG
                    if (debugLvl > 0)
                        printDebug(3, "left road (%s), new junction found with no connected sidewalk", getRoad()->getName().c_str());
                }
            }

            // Otherwise, just turn around and stay on current road (values already stored)
        }
        else
        {
            //DEBUG
            if (debugLvl > 0)
                printDebug(3, "left road (%s), no new road found", getRoad()->getName().c_str());
        }

        // Set position on new road
        if (pedState.u > getRoad()->getLength())
        {
            // Went past the end of the last road
            if (getDir() == newDir)
            {
                // Reference direction hasn't changed
                newPosition = pedState.u - getRoad()->getLength();
            }
            else
            {
                // Reference direction has changed
                newPosition = newRoad->getLength() - (pedState.u - getRoad()->getLength());
            }
            pedState.v -= pedLoc.laneSec->getDistanceToLane(getRoad()->getLength(), pedLoc.laneNum);
        }
        else if (pedState.u < 0.0)
        {
            // Went past the beginning of the last road
            if (getDir() == newDir)
            {
                // Reference direction hasn't changed
                newPosition = newRoad->getLength() + pedState.u;
            }
            else
            {
                // Reference direction has changed
                newPosition = -pedState.u;
            }
            pedState.v -= pedLoc.laneSec->getDistanceToLane(0.0, pedLoc.laneNum);
        }

        // Check whether our new position would put us on a sidewalk
        // If so, replace old road & position with the new values
        // If not, keep current values, change direction, and walk back on the current sidewalk
        LaneSection *newLaneSection = newRoad->getLaneSection(newPosition);

        // Next lane section will be this section's successor or predecessor (depending on direction)
        int newLaneNum = (getDir() > 0) ? pedLoc.laneSec->getLaneSuccessor(pedLoc.laneNum) : pedLoc.laneSec->getLanePredecessor(pedLoc.laneNum);

        bool noSidewalk = true;
        // Check whether we're still on a sidewalk
        if (newLaneSection != NULL && newLaneSection->isSidewalk(newLaneNum))
        {
            noSidewalk = false;

            // Update new lane in manager
            PedestrianManager::Instance()->changeRoad(this, getRoad(), pedLoc.laneNum, newRoad, newLaneNum);

            // Correct the lateral offset/target if reference direction has changed
            if (newDir != getDir())
            {
                // Find center of lane
                Vector2D newCenter = newLaneSection->getLaneCenter(newLaneNum, newPosition);
                // Correct values
                pedState.vStart = -1 * pedState.vStart;
                pedState.vTgt = -1 * pedState.vTgt;
                pedState.vOff = -1 * pedState.vOff;
                pedState.v = newCenter[0] + pedState.vOff;

                //DEBUG
                if (debugLvl > 0)
                    printDebug(3, "found new sidewalk (lane# %d)... different reference direction", newLaneNum);
            }

            // Replace old lane info with new lane info
            pedLoc.laneSec = newLaneSection;
            pedLoc.laneNum = newLaneNum;
            pedLoc.lane = newLaneSection->getLane(newLaneNum);

            // Replace old position with new new position
            pedState.u = newPosition;

            // Replace old road/dir with new road/dir
            pedLoc.road = newRoad;
            pedLoc.dir = newDir;
        }
        if (noSidewalk)
        {
            //DEBUG
            if (debugLvl > 0)
                printDebug(3, "no sidewalk... now turning around and remaining on lane# %d", pedLoc.laneNum);

            // No sidewalk found, so change direction and remain on current road/lane section
            pedLoc.dir = -1 * getDir();

            if (pedState.u > getRoad()->getLength())
            {
                // Move position back on road
                pedState.u = 2 * getRoad()->getLength() - pedState.u;
            }
            else if (pedState.u < 0.0)
            {
                // Move position back on road
                pedState.u = -pedState.u;
            }
            // Reset velocity to 0
            pedState.du = 0.0;
            setDuMax(pedParams.duMax, pedParams.dduMax);
        }
    }
}

/**
 * Check for entry into a new lane section due to last forward movement
 */
void Pedestrian::checkForNewLaneSection(const double dt)
{
    LaneSection *newLaneSection = getRoad()->getLaneSection(pedState.u);
    if (pedLoc.laneSec != newLaneSection)
    {
        // Pedestrian has reached a new lane section

        // Next lane section will be this section's successor or predecessor (depending on direction)
        int newLaneNum = (getDir() > 0) ? pedLoc.laneSec->getLaneSuccessor(pedLoc.laneNum) : pedLoc.laneSec->getLanePredecessor(pedLoc.laneNum);

        //DEBUG
        if (debugLvl > 0)
            printDebug(3, "entering new lane section from lane# %d, next lane is lane# %d", pedLoc.laneNum, newLaneNum);

        bool noSidewalk = true;
        // Check whether we're still on a sidewalk
        if (newLaneSection != NULL && newLaneSection->isSidewalk(newLaneNum))
        {
            noSidewalk = false;

            //DEBUG
            if (debugLvl > 0)
                printDebug(3, "found more sidewalk on lane# %d... continuing", newLaneNum);
        }

        // Update lane info for all walkable sections (don't bother otherwise, because we will be turning around)
        if (!noSidewalk)
        {
            // Update new lane in manager
            if (pedLoc.laneNum != newLaneNum)
                PedestrianManager::Instance()->changeRoad(this, getRoad(), pedLoc.laneNum, getRoad(), newLaneNum);

            // Replace old lane info with new lane info
            pedLoc.laneSec = newLaneSection;
            pedLoc.laneNum = newLaneNum;
            pedLoc.lane = newLaneSection->getLane(newLaneNum);
        }

        if (noSidewalk)
        {
            //DEBUG
            if (debugLvl > 0)
                printDebug(3, "no sidewalk... now turning around and remaining on lane# %d", pedLoc.laneNum);

            // Get back on old lane section
            pedState.u -= getDir() * pedState.du * dt;

            // No sidewalk on new lane section, so turn around and go back (also correct position to remain on old section)
            pedLoc.dir = -1 * getDir();

            // Reset velocity to 0
            pedState.du = 0.0;
            setDuMax(pedParams.duMax, pedParams.dduMax);
        }
    }
}

/**
 * Check for entry into a new crosswalk due to last forward movement
 */
void Pedestrian::checkForNewCrosswalk()
{
    Crosswalk *newCrosswalk = getRoad()->getCrosswalk(pedState.u, pedLoc.laneNum);
    if (newCrosswalk == NULL)
        pedLoc.crosswalk = NULL;
    else if (newCrosswalk != pedLoc.crosswalk)
    {
        // Just entered a new crosswalk
        pedLoc.crosswalk = newCrosswalk;

        double randNum = TrafficSimulationPlugin::plugin->getZeroOneRandomNumber();
        bool takeCrosswalk = (randNum <= newCrosswalk->getCrossProb());
        if (takeCrosswalk)
        {
            // Now crossing the street
            pedState.cross = true;
            pedState.crossProg = pedLoc.laneNum;
            pedState.crossDone = false;
            pedState.crossWaitTimer = 0.0;

            // Determine safe speed to cross road (depends on current velocity and sidewalk widths)
            double minWidth = getWalkwayWidth();

            // Find the narrowest sidewalk in this lane section
            for (int i = 1; i <= pedLoc.laneSec->getNumLanesLeft(); i++)
            {
                // Is this a sidewalk?
                if (pedLoc.laneSec != NULL && pedLoc.laneSec->isSidewalk(i))
                {
                    double thisWidth = getLaneWidth(pedLoc.laneSec, i, pedState.u);
                    if (thisWidth < minWidth)
                        minWidth = thisWidth;
                }
            }
            for (int i = 1; i <= pedLoc.laneSec->getNumLanesRight(); i++)
            {
                // Is this a sidewalk?
                if (pedLoc.laneSec != NULL && pedLoc.laneSec->isSidewalk(-1 * i))
                {
                    double thisWidth = getLaneWidth(pedLoc.laneSec, -1 * i, pedState.u);
                    if (thisWidth < minWidth)
                        minWidth = thisWidth;
                }
            }

            // Adjust speed if necessary
            pedState.crossVel = pedParams.duMax;
            if ((pedParams.duMax / minWidth) > 1.0)
            {
                pedState.crossVel = minWidth;
            }

            // Now leaving the current lane to cross the street, so remove from this lane in the manager
            PedestrianManager::Instance()->removeFromRoad(this, getRoad(), pedLoc.laneNum);

            // Attempt to enter the crosswalk right away
            pedState.crossId = Crosswalk::DONOTENTER;
            pedState.crossWaitId = Crosswalk::DONOTENTER;
            pedState.crossId = pedLoc.crosswalk->enterPedestrian(opencover::cover->frameTime());
            if (pedState.crossId == Crosswalk::DONOTENTER)
            {
                // If not possible, wait and try again later
                pedState.crossWaitId = pedLoc.crosswalk->pedestrianWaiting(opencover::cover->frameTime());
            }

            //DEBUG
            if (debugLvl > 0)
                printDebug(3, "found a crosswalk, leaving sidewalk lane# %d and crossing street at %.2f m/s", pedLoc.laneNum, pedState.crossVel);
        }
        else
        {
            //DEBUG
            if (debugLvl > 0)
                printDebug(3, "found a crosswalk, staying on sidewalk lane# %d and not crossing", pedLoc.laneNum);
        }
    }
}

/**
 * Move in the lateral (v-)direction
 */
void Pedestrian::setNewV(const double dt)
{
    if (pedState.active)
    {
        // Set acceleration
        pedState.ddv = (getPassing() != NULL) ? pedParams.ddvPass : pedParams.ddvMax;

        // Check whether we're at our target
        if ((pedState.vStart > pedState.vTgt && pedState.vOff > pedState.vTgt) || (pedState.vStart < pedState.vTgt && pedState.vOff < pedState.vTgt))
        { // Not yet at our horizontal offset target (i.e., in between start and target), so set velocity

            // Accelerate from start to halfway point, then decelerate until target

            // First part, accelerating
            // While the halfway point hasn't been reached and dvMax hasn't been reached
            if (((pedState.vStart > pedState.vTgt && pedState.vOff >= pedState.vTgt + ((pedState.vStart - pedState.vTgt) / 2)) || (pedState.vStart < pedState.vTgt && pedState.vOff <= pedState.vTgt - ((pedState.vTgt - pedState.vStart) / 2))) && ((pedState.vStart > pedState.vTgt && -pedState.dv < pedParams.dvMax) || (pedState.vStart < pedState.vTgt && pedState.dv < pedParams.dvMax)))
            {
                // Add to velocity based on acceleration
                pedState.dv += (pedState.vStart > pedState.vTgt) ? -pedState.ddv * dt : pedState.ddv * dt;

                // Remember the distance required to accelerate for decelleration phase
                pedState.vAccelDist = (pedState.vStart > pedState.vTgt) ? pedState.vStart - pedState.vOff : pedState.vOff - pedState.vStart;
            }

            // Second part, constant lateral velocity (at dvMax)
            // While we are at least accelDist away from the target
            else if ((pedState.vStart > pedState.vTgt && pedState.vOff > pedState.vTgt + pedState.vAccelDist) || (pedState.vStart < pedState.vTgt && pedState.vOff < pedState.vTgt - pedState.vAccelDist))
            {
                // No acceleration, so do nothing to velocity
            }

            // Third part, decelerating
            // While we are within accelDist away from the target
            else
            {
                pedState.dv -= (pedState.vStart > pedState.vTgt) ? -pedState.ddv * dt : pedState.ddv * dt;
            }
        }
        else if (getAvoiding() == NULL && getPassing() == NULL && getWaitingFor() == NULL && getPassedBy() == NULL)
        {
            // Got to the target, so set a new target (unless performing avoidance)
            // Generate a new random target within the center half of the lane (+/- 1/4 of the width from the lane center)
            int randNum = TrafficSimulationPlugin::plugin->getIntegerRandomNumber();
            int widthLim = (int)floor(((getWalkwayWidth()) * 0.5 * 0.25) * 100);

            // Adjust the limit based on current velocity
            if (pedState.du < 1.5)
                widthLim = (int)floor(((double)(widthLim)) * pedState.du / 1.5);
            if (widthLim > 0)
            {
                double tgt = ((double)(randNum % (widthLim + 1))) / 100;

                // Also randomize choice of positive/negative
                tgt = (TrafficSimulationPlugin::plugin->getZeroOneRandomNumber() <= 0.5) ? tgt : -tgt;

                //DEBUG
                if (debugLvl > 0)
                    printDebug(6, "lateral target of %.2f reached; new target at %.2f", pedState.vTgt, tgt);

                // Set this target
                setVTarget(tgt);
            }
        }
        pedState.vOff += pedState.dv * dt;
    }

    // Set lateral position (if u is off the road, use closest on-road value to calculate v... u will be corrected later when checking for sidewalk)
    double u = pedState.u;
    if (u < 0.0)
        u = 0.0;
    else if (u > getRoad()->getLength())
        u = getRoad()->getLength();
    Vector2D laneCenter = pedLoc.laneSec->getLaneCenter(pedLoc.laneNum, u);
    pedState.v = laneCenter[0] + pedState.vOff;
}

/**
 * If taking a crosswalk, handle movement across the street
 */
void Pedestrian::crossStreet(const double dt)
{
    if (!pedState.crossDone)
    {
        // Decelerate du until it reaches 0
        if (pedState.du > 0)
        {
            // Not yet at 0, decrease
            pedState.du -= pedParams.duMax * dt;
            if (pedState.du < 0)
                pedState.du = 0.0;
        }

        // Accelerate dv
        //  in appropriate direction, (i.e., if laneNum is positive, dv should be negative to reach the next more-negative sidewalk)
        if (pedLoc.laneNum > 0)
        {
            // Positive laneNum, need to have a negative dv to cross the street
            if (pedState.dv > -pedState.crossVel)
            {
                // Not yet at target, decrease
                pedState.dv -= pedState.crossVel * dt;
                if (pedState.dv < -pedState.crossVel)
                    pedState.dv = -pedState.crossVel;
            }
        }
        else if (pedLoc.laneNum < 0)
        {
            // Negative laneNum, need to have a positive dv to cross the street
            if (pedState.dv < pedState.crossVel)
            {
                // Not yet at target, increase
                pedState.dv += pedState.crossVel * dt;
                if (pedState.dv > pedState.crossVel)
                    pedState.dv = pedState.crossVel;
            }
        }
    }

    // Move according to new velocities
    pedState.u += getDir() * pedState.du * dt;
    pedState.s += pedState.du * dt;
    pedState.v += pedState.dv * dt;

    // Update geometry's animation
    if (fabs(pedState.dv) < 1.0e-8)
    {
        geometry->setWalkingSpeed(pedState.du);
    }
    else if (fabs(pedState.du) < 1.0e-8)
    {
        geometry->setWalkingSpeed(pedState.dv);
    }
    else
    {
        geometry->setWalkingSpeed(sqrt(pedState.dv * pedState.dv + pedState.du * pedState.du));
    }

    // Check pedestrian's position on the street
    int laneNum = pedLoc.laneSec->searchLane(pedState.u, pedState.v);
    if (laneNum != pedLoc.laneNum && laneNum != pedState.crossProg && laneNum != Lane::NOLANE)
    {
        // If the lane is not (a) the starting lane or (b) no lane, and (c) has not yet been visited...
        Lane *lane = pedLoc.laneSec->getLane(laneNum);
        Lane::LaneType laneType = Lane::NONE;
        if (lane != NULL)
            laneType = lane->getLaneType();
        if (laneType != Lane::SIDEWALK && laneType != Lane::NONE)
        {
            // Reached a driving lane or a misc. center lane
            // Check whether we're allowed to cross
            if (pedState.crossId != Crosswalk::DONOTENTER)
            {
                // Wait for at least a second before crossing
                if (pedState.crossWaitTimer < 1.0)
                {
                    // Undo last movement
                    pedState.u -= getDir() * pedState.du * dt;
                    pedState.s -= pedState.du * dt;
                    pedState.v -= pedState.dv * dt;

                    // Walking speed is 0
                    geometry->setWalkingSpeed(0.0);

                    if (pedState.crossWaitTimer < 1.0e-8)
                    {
                        geometry->executeLook(0.9 + 0.3 * TrafficSimulationPlugin::plugin->getZeroOneRandomNumber());

                        //DEBUG
                        if (debugLvl > 0)
                            printDebug(6, "reached a driving lane, waiting 1 second (look both ways) before entering the street");
                    }
                    pedState.crossWaitTimer += dt;
                }
                else
                {
                    // If done waiting, update lane info and proceed
                    pedState.crossProg = laneNum;

                    //DEBUG
                    if (debugLvl > 0)
                        printDebug(6, "crossing street from lane# %d, reached a%s at lane# %d", pedLoc.laneNum, ((laneType == Lane::DRIVING) ? " driving lane" : "n undefined crosswalk lane"), laneNum);
                }
            }
            else
            {
                // Undo last movement
                pedState.u -= getDir() * pedState.du * dt;
                pedState.s -= pedState.du * dt;
                pedState.v -= pedState.dv * dt;

                // Walking speed is 0
                geometry->setWalkingSpeed(0.0);

                // Not yet allowed to enter the crosswalk (should have a waitId)
                if (pedState.crossWaitId == Crosswalk::DONOTENTER)
                {
                    // Get a waitId if none is assigned
                    pedState.crossWaitId = pedLoc.crosswalk->pedestrianWaiting(opencover::cover->frameTime());
                }

                // Attempt to get a crossId (see if it's possible to enter the crosswalk)
                pedState.crossId = pedLoc.crosswalk->enterPedestrian(opencover::cover->frameTime());
                if (pedState.crossId != Crosswalk::DONOTENTER)
                {
                    // Allowed to enter the crosswalk, so stop waiting
                    if (pedLoc.crosswalk->pedestrianDoneWaiting(pedState.crossWaitId, opencover::cover->frameTime()) == pedState.crossWaitId)
                        pedState.crossWaitId = Crosswalk::DONOTENTER;
                }
            }
        }
        else if (laneType == Lane::SIDEWALK)
        {
            // Reached the opposite sidewalk, fix acceleration, add pedestrian to new roadside, and stop crossing
            if (!pedState.crossDone)
            {
                if (pedLoc.crosswalk->exitPedestrian(pedState.crossId, opencover::cover->frameTime()) == pedState.crossId)
                    pedState.crossId = Crosswalk::DONOTENTER;
                pedState.crossDone = true;

                //DEBUG
                if (debugLvl > 0)
                    printDebug(4, "reached a sidewalk on this crosswalk, beginning to move forward");
            }
            if (pedLoc.laneNum > 0 ? pedState.dv < 0 : pedState.dv > 0)
            {
                // increase du up to duMax
                if (pedState.du < pedParams.duMax)
                {
                    pedState.du += pedParams.duMax * dt;
                    if (pedState.du > pedParams.duMax)
                        pedState.du = pedParams.duMax;
                }

                // decrease dv down to 0
                // If lane width is too narrow, increase deceleration appropriately
                if (pedLoc.laneNum > 0)
                {
                    // Started crossing from a positive-numbered lane
                    pedState.dv += pedState.crossVel * dt;
                    if (pedState.dv > 0)
                        pedState.dv = 0;
                }
                else
                {
                    // Started crossing from a negative-numbered lane
                    pedState.dv -= pedState.crossVel * dt;
                    if (pedState.dv < 0)
                        pedState.dv = 0;
                }
            }
            else
            {
                // Done decelerating, done crossing, continue on new lane/roadside
                pedState.cross = false;

                // Update new lane info
                pedLoc.laneNum = laneNum;
                pedLoc.lane = lane;

                // Update this pedestrian's road/lane in the manager
                PedestrianManager::Instance()->addToRoad(this, getRoad(), laneNum);

                // Reset velocities
                pedState.duStart = pedState.du;
                pedState.duTgt = pedParams.duMax;

                // Set vTgt to current offset, forcing a new target to be set
                Vector2D laneCenter = pedLoc.laneSec->getLaneCenter(pedLoc.laneNum, pedState.u);
                pedState.vOff = pedState.v - laneCenter[0];
                setVTarget(pedState.vOff);

                //DEBUG
                if (debugLvl > 0)
                    printDebug(3, "done crossing street, reached sidewalk at lane# %d and now moving forward", pedLoc.laneNum);
            }
        }
        else // laneType == Lane::NONE
        {
            // Remove from the pedestrian system
            PedestrianManager::Instance()->scheduleRemoval(this);

            //DEBUG
            printDebug(0, "error crossing street, left the crosswalk at lane# %d, trying to return to lane# %d", laneNum, -pedLoc.laneNum);
        }
    }
    else
    {
        // If the lane is (a) the starting lane or (b) no lane, or (c) has already been visited...
    }
}

/**
 * If searching for a sidewalk, decrease vOffset until a sidewalk is reached
 */
void Pedestrian::findSidewalk(const double dt)
{
    // Determine "safe" velocity so that we don't overshoot the lane when we get to it
    double safeVel = pedParams.duMax;
    if ((pedParams.duMax / getWalkwayWidth()) > 1.0)
    {
        safeVel = getWalkwayWidth();
    }

    // If we are still just moving laterally...
    if (!pedState.foundSidewalk)
    {
        // No forward velocity
        pedState.du = 0;

        // Accelerate dv in appropriate direction
        if (pedState.vStart > 0)
        {
            // Decreasing vOff to 0
            if (pedState.dv > -safeVel)
            {
                // Not yet at target, decrease
                pedState.dv -= safeVel * dt;
                if (pedState.dv < -safeVel)
                    pedState.dv = -safeVel;
            }
        }
        else if (pedState.vStart < 0)
        {
            // Increasing vOff to 0
            if (pedState.dv < safeVel)
            {
                // Not yet at target, increase
                pedState.dv += safeVel * dt;
                if (pedState.dv > safeVel)
                    pedState.dv = safeVel;
            }
        }
    }

    // Move according to new velocity
    pedState.u += getDir() * pedState.du * dt;
    pedState.s += pedState.du * dt;
    pedState.v += pedState.dv * dt;
    pedState.vOff += pedState.dv * dt;

    // Update geometry's animation
    if (fabs(pedState.dv) < 1.0e-8)
    {
        geometry->setWalkingSpeed(pedState.du);
    }
    else if (fabs(pedState.du) < 1.0e-8)
    {
        geometry->setWalkingSpeed(pedState.dv);
    }
    else
    {
        geometry->setWalkingSpeed(sqrt(pedState.dv * pedState.dv + pedState.du * pedState.du));
    }

    if (pedState.foundSidewalk)
    {
        // Fix velocities and then stop searching
        if (pedState.vStart > 0 ? pedState.dv < 0 : pedState.dv > 0)
        {
            // increase du up to duMax
            if (pedState.du < pedParams.duMax)
            {
                pedState.du += pedParams.duMax * dt;
                if (pedState.du > pedParams.duMax)
                    pedState.du = pedParams.duMax;
            }

            // decrease dv down to 0
            // If lane width is too narrow, increase deceleration appropriately
            if (pedState.vStart > 0)
            {
                // Negative dv, so increase
                pedState.dv += safeVel * dt;
                if (pedState.dv > 0)
                    pedState.dv = 0;
            }
            else
            {
                // Positive dv, so decrease
                pedState.dv -= safeVel * dt;
                if (pedState.dv < 0)
                    pedState.dv = 0;
            }
        }
        else
        {
            // Done decelerating, done searching, continue on lane
            pedState.searchSidewalk = false;

            //DEBUG
            if (debugLvl > 0)
                printDebug(4, "done decelerating onto sidewalk, continuing on lane# %d", pedLoc.laneNum);
        }
    }
    else
    {
        // Check pedestrian's position on the street
        int laneNum = pedLoc.laneSec->searchLane(pedState.u, pedState.v);
        if (laneNum != Lane::NOLANE)
        {
            // If we are on some sort of lane...
            Lane *lane = pedLoc.laneSec->getLane(laneNum);
            Lane::LaneType laneType = Lane::NONE;
            if (lane != NULL)
                laneType = lane->getLaneType();
            if (laneType == Lane::SIDEWALK)
            {
                // Found the sidewalk, stop searching and start walking along it
                pedState.foundSidewalk = true;

                // Add back to road
                PedestrianManager::Instance()->addToRoad(this, getRoad(), pedLoc.laneNum);

                //DEBUG
                if (debugLvl > 0)
                    printDebug(3, "reached the starting sidewalk on lane# %d, beginning to move forward", pedLoc.laneNum);
            }
        }
    }
}

/**
 * Walk to the given vOffset, then this pedestrian will be removed
 */
void Pedestrian::leavingSidewalk(const double dt)
{
    // Remove from road as soon as leaving begins
    if (isOnRoad())
        PedestrianManager::Instance()->removeFromRoad(this, getRoad(), pedLoc.laneNum);

    // Accelerate dv in appropriate direction if necessary
    if (pedState.vTgt > pedState.vStart)
    {
        // Increasing vOff to vTgt
        if (pedState.dv < pedParams.duMax)
        {
            // Not yet at target, decrease
            pedState.dv += pedParams.duMax * dt;
            if (pedState.dv > pedParams.duMax)
                pedState.dv = pedParams.duMax;
        }
    }
    else if (pedState.vTgt < pedState.vStart)
    {
        // Decreasing vOff to vTgt
        if (pedState.dv > -pedParams.duMax)
        {
            // Not yet at target, increase
            pedState.dv -= pedParams.duMax * dt;
            if (pedState.dv < -pedParams.duMax)
                pedState.dv = -pedParams.duMax;
        }
    }

    // Decelerate du to 0
    if (pedState.du > 0)
    {
        pedState.du -= pedParams.duMax * dt;
        if (pedState.du < 0)
            pedState.du = 0;
    }

    // Move according to new velocity
    pedState.u += getDir() * pedState.du * dt;
    pedState.s += pedState.du * dt;
    pedState.v += pedState.dv * dt;
    pedState.vOff += pedState.dv * dt;

    // Update geometry's animation
    if (fabs(pedState.dv) < 1.0e-8)
    {
        geometry->setWalkingSpeed(pedState.du);
    }
    else if (fabs(pedState.du) < 1.0e-8)
    {
        geometry->setWalkingSpeed(pedState.dv);
    }
    else
    {
        geometry->setWalkingSpeed(sqrt(pedState.dv * pedState.dv + pedState.du * pedState.du));
    }

    // Get to vTgt, then disappear
    if (pedState.leavingDone)
    {
        //DEBUG
        if (debugLvl > 0)
            printDebug(4, "reached vTgt (off %.2f > tgt %.2f) and done leaving the sidewalk, now being removed", pedState.vOff, pedState.vTgt);

        // Remove from the pedestrian system
        PedestrianManager::Instance()->scheduleRemoval(this);
    }
    else if ((pedState.vTgt > pedState.vStart && pedState.vOff > pedState.vTgt) || (pedState.vTgt < pedState.vStart && pedState.vOff < pedState.vTgt))
    {
        //DEBUG
        if (debugLvl > 0)
            printDebug(3, "done leaving sidewalk (vStart=%.2f, vTgt=%.2f, vOff=%.2f)", pedState.vStart, pedState.vTgt, pedState.vOff);

        pedState.leavingDone = true;
    }
}

/**
 * Face the geometry forward in the current direction of movement (take into account v- and u-velocities)
 */
void Pedestrian::setNewHdg()
{
    // Calculate heading based on current longitudinal and lateral velocities
    if (fabs(pedState.dv) < 1.0e-8 && fabs(pedState.du) < 1.0e-8)
    {
        // No velocity, face forward and stand still
        pedState.hdg = M_PI / 2 * getDir();
    }
    else if (fabs(pedState.dv) < 1.0e-8)
    {
        // Forward velocity, face forward or backward depending on direction
        pedState.hdg = M_PI / 2 * getDir();
    }
    else if (fabs(pedState.du) < 1.0e-8)
    {
        // Lateral velocity, face left or right depending on pos/neg
        pedState.hdg = (pedState.dv < 0) ? 0 : M_PI;
    }
    else
    {
        // Lateral&Forward velocity, so diagonal heading
        double dw = sqrt(pedState.du * pedState.du - pedState.dv * pedState.dv);
        if (dw != dw)
            dw = 0;
        else
        {
            pedState.hdg = M_PI / 2 * getDir() + getDir() * atan(pedState.dv / dw);
            if (pedState.hdg != pedState.hdg)
            {
                pedState.hdg = M_PI / 2 * getDir();
            }
        }
    }
}

/**
 * Update the pedestrian's geometry based on current position and road transform
 */
void Pedestrian::updateGeometry()
{
    if (pedState.u < 0.0)
    {
        printDebug(3, "did not update geometry! current position is %.2f", pedState.u);
    }
    else
    {
        pedTrans = getRoad()->getRoadTransform(pedState.u, pedState.v);
        geometry->setTransform(pedTrans, (pedState.hdg + pedParams.headingAdj * M_PI));
    }
}

/**
 * Print a status/error message
 */
void Pedestrian::printDebug(int lvl, const char *msg, ...)
{
    /**
    * Debug levels are (roughly):
    *   0) errors
    *   1) creation and removal
    *   2) active and inactive
    *   3) routing decisions (roads, lane sections, crosswalks)
    *   4) passing, avoiding, aux routing decisions
    *   5) velocity settings (initial, passing, avoiding)
    *   6) target accomplishments/settings (velocity, vOff, crosswalk)
    */

    // If this message is at/below the current debug level, print it
    if (lvl <= debugLvl)
    {
        // Print pedestrian's name
        fprintf(stderr, " Pedestrian '%s': ", getName());

        va_list list;
        va_start(list, msg);

        // Print the list
        vfprintf(stderr, msg, list);

        va_end(list);

        // Print a newline
        fprintf(stderr, "\n");
    }
}
