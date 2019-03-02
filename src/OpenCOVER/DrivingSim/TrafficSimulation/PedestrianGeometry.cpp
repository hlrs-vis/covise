/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PedestrianGeometry.h"

/**
 * Construct a new pedestrian geometry object
 */
PedestrianGeometry::PedestrianGeometry(std::string &name, std::string &modelFile, double scale, double lod, const PedestrianAnimations &a, osg::Group *group)
    : geometryName(name)
    , pedGroup(group)
    , anim(a)
    , rangeLOD(lod)
    , timeFactorScale(1.0)
    , animOffset(0.0)
    , currentSpeed(-1.0)
    , lastSpeed(-1.0)
{
    // Create a matrix for scale
    mScale.makeScale(scale, scale, scale);

    // Create a new transform, add it to the group
    pedTransform = new osg::MatrixTransform();
    pedTransform->setName(name);
    pedGroup->addChild(pedTransform);

    // Create a new LOD, set range, and add it to the transform
    pedLOD = new osg::LOD();
    pedTransform->addChild(pedLOD);
    pedLOD->setRange(0, 0.0, 2*lod/scale); // *2 == render them twice as far as they are animated

    // Create a new instance of the core model, add it to the LOD
    pedModel = new osgCal::Model();
    meshAdder = new osgCal::DefaultMeshAdder;
    osg::ref_ptr<osgCal::CoreModel> coreModel = PedestrianFactory::Instance()->getCoreModel(modelFile);
    pedModel->load(coreModel, meshAdder);
    pedLOD->addChild(pedModel);

    // Make active (set default mask)
    mask = pedTransform->getNodeMask() & ~(opencover::Isect::Intersection | opencover::Isect::Collision | opencover::Isect::Walk);
    pedTransform->setNodeMask(mask);
}
PedestrianGeometry::~PedestrianGeometry()
{
    removeFromSceneGraph();
}

/**
 * Remove the geometry from the scenegraph (detach the top-level geometry node (MatrixTransform) from the PedestrianGroup)
 */
void PedestrianGeometry::removeFromSceneGraph()
{
    pedGroup->removeChild(pedTransform);
}

void PedestrianGeometry::update(double dt)
{
	pedModel->update(dt);
}

/**
 * Adjust the geometry's animation settings to match the given speed, according to the geometry's animation mapping
 */
void PedestrianGeometry::setWalkingSpeed(double speed)
{
    // Check whether there was any change in velocity
    lastSpeed = currentSpeed;
    currentSpeed = speed;
    if (currentSpeed != lastSpeed)
    {
        // Only allow positive speeds
        if (speed < 0.0)
            speed = -speed;

        // Avoid abrupt motion changes
        double blendTime = 0.1;

        // Will be blending four animations
        double idleAmt = 0.0;
        double slowAmt = 0.0;
        double walkAmt = 0.0;
        double jogAmt = 0.0;
        double effVel = speed;

        // Clear any cycles that are out of range
        //  and blend any cycles that are in range
        if (speed <= 0.0001)
        {
            pedModel->clearCycle(anim.slowIdx, blendTime);
            pedModel->clearCycle(anim.walkIdx, blendTime);
            pedModel->clearCycle(anim.jogIdx, blendTime);

            idleAmt = 1.0;
            double scale = 1.0; // Play idle at full speed, because the animation should have vel=0m/s

            pedModel->blendCycle(anim.idleIdx, idleAmt, blendTime);
            if (!PedestrianUtils::floatEq(timeFactorScale, scale))
            {
                timeFactorScale = scale;
                pedModel->setTimeFactor(timeFactorScale);
            }
        }
        else if (speed <= anim.slowVel)
        {
            pedModel->clearCycle(anim.walkIdx, blendTime);
            pedModel->clearCycle(anim.jogIdx, blendTime);

            idleAmt = ((anim.slowVel - anim.idleVel) - (speed - anim.idleVel)) / (anim.slowVel - anim.idleVel);
            if (idleAmt < 0.0)
                idleAmt = 0.0;
            if (idleAmt > 1.0)
                idleAmt = 1.0;
            slowAmt = 1.0 - idleAmt;
            effVel = anim.idleVel * idleAmt + anim.slowVel * slowAmt; // should equal 'speed'
            double scale = speed / effVel; // should equal 1.0f

            pedModel->blendCycle(anim.idleIdx, idleAmt, blendTime);
            pedModel->blendCycle(anim.slowIdx, slowAmt, blendTime);
            if (!PedestrianUtils::floatEq(timeFactorScale, scale))
            {
                timeFactorScale = scale;
                pedModel->setTimeFactor(timeFactorScale);
            }
        }
        else if (speed <= anim.walkVel)
        {
            pedModel->clearCycle(anim.idleIdx, blendTime);
            pedModel->clearCycle(anim.jogIdx, blendTime);

            slowAmt = ((anim.walkVel - anim.slowVel) - (speed - anim.slowVel)) / (anim.walkVel - anim.slowVel);
            if (slowAmt < 0.0)
                slowAmt = 0.0;
            if (slowAmt > 1.0)
                slowAmt = 1.0;
            walkAmt = 1.0 - slowAmt;
            effVel = anim.slowVel * slowAmt + anim.walkVel * walkAmt;
            double scale = speed / effVel;

            pedModel->blendCycle(anim.slowIdx, slowAmt, blendTime);
            pedModel->blendCycle(anim.walkIdx, walkAmt, blendTime);
            if (!PedestrianUtils::floatEq(timeFactorScale, scale))
            {
                timeFactorScale = scale;
                pedModel->setTimeFactor(timeFactorScale);
            }
        }
        else if (speed <= anim.jogVel)
        {
            pedModel->clearCycle(anim.idleIdx, blendTime);
            pedModel->clearCycle(anim.slowIdx, blendTime);

            walkAmt = ((anim.jogVel - anim.walkVel) - (speed - anim.walkVel)) / (anim.jogVel - anim.walkVel);
            if (walkAmt < 0.0)
                walkAmt = 0.0;
            if (walkAmt > 1.0)
                walkAmt = 1.0;
            jogAmt = 1.0 - walkAmt;
            effVel = anim.walkVel * walkAmt + anim.jogVel * jogAmt;
            double scale = speed / effVel;

            pedModel->blendCycle(anim.walkIdx, walkAmt, blendTime);
            pedModel->blendCycle(anim.jogIdx, jogAmt, blendTime);
            if (!PedestrianUtils::floatEq(timeFactorScale, scale))
            {
                timeFactorScale = scale;
                pedModel->setTimeFactor(timeFactorScale);
            }
        }
        else
        {
            pedModel->clearCycle(anim.idleIdx, blendTime);
            pedModel->clearCycle(anim.slowIdx, blendTime);
            pedModel->clearCycle(anim.walkIdx, blendTime);

            jogAmt = 1.0;
            double scale = speed / anim.jogVel;

            pedModel->blendCycle(anim.jogIdx, jogAmt, blendTime);
            if (!PedestrianUtils::floatEq(timeFactorScale, scale))
            {
                timeFactorScale = scale;
                pedModel->setTimeFactor(timeFactorScale);
            }
        }

        // Check that amounts totals, scale, and effective velocity are correct
        double totalAmt = idleAmt + slowAmt + walkAmt + jogAmt;
        if (!PedestrianUtils::floatEq(totalAmt, 1.0))
            fprintf(stderr, " Pedestrian '%s': animation total equals %.2f, not 1.0\n", geometryName.c_str(), totalAmt);
        if (!PedestrianUtils::floatEq(idleAmt, 1.0) && !PedestrianUtils::floatEq(jogAmt, 1.0))
        {
            if (!PedestrianUtils::floatEq(timeFactorScale, 1.0))
                fprintf(stderr, " Pedestrian '%s': velocity is %.2f but time factor scale equals %.2f, not 1.0\n", geometryName.c_str(), speed, timeFactorScale);
        }
        if (!PedestrianUtils::floatEq(effVel, speed))
        {
            fprintf(stderr, " Pedestrian '%s': effective velocity (%.2f) does not match speed (%.2f)\n", geometryName.c_str(), effVel, speed);
        }
    }
}

/**
 * Set the geometry to match the given road transformation and heading
 */
void PedestrianGeometry::setTransform(Transform &roadTransform, double heading)
{
    Quaternion qzaaa(heading, Vector3D(0, 0, 1));
    Quaternion q = roadTransform.q() * qzaaa;

    // Calculate Euler angle for heading (ignore pitch and roll)
    double qx = q.x();
    double qy = q.y();
    double qz = q.z();
    double qw = q.w();
    double sqx = qx * qx;
    double sqy = qy * qy;
    double sqz = qz * qz;
    double sqw = qw * qw;
    double term1 = 2 * (qx * qy + qw * qz);
    double term2 = sqw + sqx - sqy - sqz;
    //double term3 = -2*(qx*qz-qw*qy);
    //double term4 = 2*(qw*qx+qy*qz);
    //double term5 = sqw - sqx - sqy + sqz;
    double h = atan2(term1, term2);
    //double p = atan2(term4, term5);
    //double r = asin(term3);

    // Generate and set a transformation matrix
    osg::Matrix m;
    m.makeRotate(h, 0, 0, 1);
    m.setTrans(roadTransform.v().x(), roadTransform.v().y(), roadTransform.v().z());

    // Set the transform
    pedTransform->setMatrix(mScale * m);
}

/**
 * Returns TRUE if the distance between the given point and the geometry's transform is within the LOD range; FALSE otherwise
 */
bool PedestrianGeometry::isGeometryWithinLOD()
{
    return isGeometryWithinRange(rangeLOD);
}

/**
 * Returns TRUE if the geometry's transform is within the given range; FALSE otherwise
 */
bool PedestrianGeometry::isGeometryWithinRange(const double r) const
{
    // Find viewer's position and geometry's position in object coordinates
    osg::Vec3d viewerPosWld = opencover::cover->getViewerMat().getTrans();
    osg::Vec3d viewPos = viewerPosWld * opencover::cover->getInvBaseMat();
    osg::Vec3d geomPos = pedTransform->getMatrix().getTrans();

	double distSq = (geomPos - viewPos).length2();
    double rangeSq = r * r;

    // Compare it to the square of the LOD range; if it's less, then it's within the LOD range
    if (distSq <= rangeSq)
        return true;
    else
        return false;
}

/**
 * Perform the one-time action to "look both ways before crossing the street" (i.e., turn head left and right)
 */
void PedestrianGeometry::executeLook(double factor)
{
    executeAction(anim.lookIdx, factor);
}

/**
 * Perform the one-time action to "wave" (e.g., when approaching a fellow pedestrian)
 */
void PedestrianGeometry::executeWave(double factor)
{
    executeAction(anim.waveIdx, factor);
}

/**
 * Perform a one-time action, where idx is its position in the Cal3d .cfg file's animation list (first item has idx 0)
 */
void PedestrianGeometry::executeAction(int idx, double factor)
{
    factor = factor < 0 ? -factor : factor;
    if (idx >= 0)
    {
        if (PedestrianUtils::floatEq(factor, 1.0))
            pedModel->executeAction(idx, factor);
        else
            pedModel->executeAction(idx, 0.0, 0.0, 1.0, false, factor);
    }
}
