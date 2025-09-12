/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TacxFTMS.h"

#include <OpenVRUI/osg/mathUtils.h>  //for MAKE_EULER_MAT
#include <config/CoviseConfig.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRCollaboration.h>
#include <cover/coVRMSController.h>
#include <cover/input/deviceDiscovery.h>
#include <cover/input/input.h>
#include <util/UDPComm.h>
#include <util/byteswap.h>
#include <util/unixcompat.h>

#include <osg/LineSegment>
#include <osgUtil/IntersectionVisitor>
#include <osgUtil/LineSegmentIntersector>

#include "cover/coIntersection.h"
using namespace opencover;

// To be added
// static float zeroAngle = _;

TacxFTMS::TacxFTMS()
: coVRPlugin(COVER_PLUGIN_NAME)
, udpNeo(NULL)
, udpAlpine(NULL)
, coVRNavigationProvider("TacxFTMS", this)
{
    ftmsData.speed = 0.0;
    ftmsData.cadence = 0.0;
    ftmsData.power = 0.0;

    ftmsControl.wind_speed = 0.0;
    ftmsControl.grade = 0.0;
    ftmsControl.crr = 0.0033; // random default value in the range of road bike
    ftmsControl.cw = 0.49;  //  random default value for average rider 
    ftmsControl.weight = 72.0;

    stepSizeUp = 2000;
    stepSizeDown = 2000;

    coVRNavigationManager::instance()->registerNavigationProvider(this);

    opencover::Input::instance()->discovery()->deviceAdded.connect(&TacxFTMS::addDevice, this);
}

TacxFTMS::~TacxFTMS() {
    coVRNavigationManager::instance()->unregisterNavigationProvider(this);
    running = false;

    delete udpNeo;
    delete udpAlpine;
}

bool TacxFTMS::init() {
    std::cerr << "Init FTMS started";
    float floorHeight = VRSceneGraph::instance()->floorHeight();

    float x =
        configFloat("Position", "x", 0)->value();
    float y =
        configFloat("Position", "y", 0)->value();
    float z = 
        configFloat("Position", "z", floorHeight)->value();
    float h =
        configFloat("Orientation", "h", 0)->value();
    float p =
        configFloat("Orientation", "p", 0)->value();
    float r =
        configFloat("Orientation", "r", 0)->value();

    MAKE_EULER_MAT(TacxFTMSPos, h, p, r);
    TacxFTMSPos.postMultTranslate(osg::Vec3(x, y, z));


    /*std::cerr << "TacxFTMS config: UDP: serverHost: " << host
              << ", localPort: " << localPort << ", serverPort: " << serverPort
              << std::endl;*/
    bool supportedDeviceFound = false;

    bool ret = false;

    if (coVRMSController::instance()->isMaster()) {
        std::string host = "";
        for (const auto& i :
             opencover::Input::instance()->discovery()->getDevices()) {
                std::cerr << "Devicename found" << i->deviceName << std::endl;
            if (i->deviceName == "TacxNeo") {
                supportedDeviceFound = true;
            }
            else if (i->deviceName == "Alpine")
            {
                supportedDeviceFound = true;
            }
        }
        ret = supportedDeviceFound;
        coVRMSController::instance()->sendSlaves(&ret, sizeof(ret));
    }
    else
    {
        coVRMSController::instance()->readMaster(&ret, sizeof(ret));
    }
    std::cerr << "Init FTMS done";

    return ret;
}

void TacxFTMS::addDevice(const opencover::deviceInfo *i)
{
    //const std::string host = configString("TacxFTMS", "severHost", "192.168.178.36")->value();
    unsigned short serverPortNeo = configInt("TacxFTMS", "serverPort", 31319)->value();
    unsigned short localPortNeo = configInt("TacxFTMS", "localPort", 31322)->value();

    unsigned short serverPortAlpine = configInt("Alpine", "serverPort", 31319)->value();
    unsigned short localPortAlpine = configInt("Alpine", "localPort", 31328)->value();


    if (coVRMSController::instance()->isMaster())
    {
        std::string host = "";
        std::cerr << "Devicename found" << i->deviceName << std::endl;
        if (i->deviceName == "TacxNeo")
        {
            host = i->address;
            std::cerr << "TacxFTMS config: UDP: TacxHost: " << host << std::endl;
            udpNeo = new UDPComm(host.c_str(), serverPortNeo, localPortNeo);
            if (!udpNeo->isBad())
            {
                ftmsfound = true;
                start();
            }
            else
            {
                std::cerr << "TacxFTMS: failed to open local UDP port" << localPortNeo << std::endl;
                ftmsfound = false;
            }
        }
        else if (i->deviceName == "Alpine")
        {
            host = i->address;
            std::cerr << "TacxFTMS config: UDP: AlpineHost: " << host << std::endl;
            udpAlpine = new UDPComm(host.c_str(), serverPortAlpine, localPortAlpine);
            if (!udpAlpine->isBad())
            {
                alpinefound = true;
                start();
            }
            else
            {
                std::cerr << "Alpine: failed to open local UDP port" << localPortAlpine << std::endl;
                alpinefound = false;
            }
        }
        ret = ftmsfound || alpinefound;
        coVRMSController::instance()->sendSlaves(&ret, sizeof(ret));
    }
    else
    {
        coVRMSController::instance()->readMaster(&ret, sizeof(ret));
    }
    ftmsfound = coVRMSController::instance()->syncBool(ftmsfound);
    alpinefound = coVRMSController::instance()->syncBool(alpinefound);
}

bool TacxFTMS::update() {
    if (isEnabled()) {
        if (coVRMSController::instance()->isMaster()) {
            float speed = getSpeed();
            //fprintf(stderr, "speed: %f\n", speed);

            double dT = cover->frameDuration();

            TransformMat =
                VRSceneGraph::instance()->getTransform()->getMatrix();

            float grade = getGrade();

            if (fabs(speed) < 0.00001) {
                speed = 0;
            }

            float s = speed * dT;
            // fprintf(stderr, "Displacement: %lf   Speed: %lf    dt: %lf
            // Y-acceleration: %lf\n", s, speed, dT, a);

            osg::Vec3 V(0, s, 0);
            float rotAngle = 0.0;
            float wheelAngle = getAngle() / 10.0;
            //fprintf(stderr, "wheelAngle: %f\n", wheelAngle);

            if (fabs(s) < 0.001 || fabs(wheelAngle) < 0.001) {
                rotAngle = 0;
            } else {
                float r = tan(M_PI_2 -
                              wheelAngle * 0.2 / (((fabs(speed) * 0.2) + 1))) *
                          wheelBase;
                float u = 2.0 * r * M_PI;
                rotAngle = (s / u) * 2.0 * M_PI;
                V[1] = r * sin(rotAngle);
                V[0] = (r - r * cos(rotAngle));
            }

            osg::Matrix relTrans;
            osg::Matrix relRot;
            relRot.makeRotate(-rotAngle, 0, 0, 1);
            relTrans.makeTranslate(
                V * -1000);  // m to mm (move world in the opposite direction

            auto mat = getMatrix();

            TransformMat = mat * relTrans * relRot;

            coVRMSController::instance()->sendSlaves((char*)TransformMat.ptr(),
                                                     sizeof(TransformMat));
        } else {
            coVRMSController::instance()->readMaster((char*)TransformMat.ptr(),
                                                     sizeof(TransformMat));
        }
        VRSceneGraph::instance()->getTransform()->setMatrix(TransformMat);
        coVRCollaboration::instance()->SyncXform();
    }
    return false;
}


float TacxFTMS::getGrade() {
    osg::Vec3d forward{TransformMat(1, 0), TransformMat(1, 1), TransformMat(1, 2)};
    forward.normalize();

    osg::Vec3d forwardXY{forward.x(), forward.y(), 0.0};
    double run = forwardXY.length();
    double rise = forward.z();

    float grade = 0.0f;
    if (run > 1e-6) {
        grade = (rise / run) * 100.0f;
    }

    //fprintf(stderr, "grade: %.2f%%\n", grade);
    return grade;
}

void TacxFTMS::setEnabled(bool flag) {
    coVRNavigationProvider::setEnabled(flag);
    if (udpNeo) {
        if (flag) {
            udpNeo->send("start");
        } else {
            udpNeo->send("stop");
        }
    }
    if (udpAlpine) {
        if (flag) {
            udpAlpine->send("start");
        } else {
            udpAlpine->send("stop");
        }
    }
    // WakeUp TacxFTMS
    Initialize();
}

void TacxFTMS::run() {
    running = true;
    doStop = false;
    if (coVRMSController::instance()->isMaster()) {
        while (running && !doStop) {
            this->updateThread();
        }
    }
}

void TacxFTMS::updateThread() {
    if (udpNeo) {
        char tmpBuf[10000];
        int status;
        status = udpNeo->receive(&tmpBuf, 10000);

        if (status == -1) {
            if (isEnabled())  // otherwise we are not supposed to receive
                              // anything
            {
                std::cerr << "TacxFTMS::update: error while reading Neo data"
                          << std::endl;
                if (udpNeo)  // try to wake up the trainer (if the first start UDP
                          // message was lost)
                    udpNeo->send("start");
            }
        } else if (status >= sizeof(FTMSBikeData)) {
            if (!isEnabled()) {
                if (udpNeo)  // still receiving data, send stop
                    udpNeo->send("stop");
            }
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
            memcpy(&ftmsData, tmpBuf, sizeof(FTMSBikeData));

            sendIndoorBikeSimulationParameters();
        } else if (status >= sizeof(AlpineData)) {

            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
            memcpy(&alpineData, tmpBuf, sizeof(AlpineData));
        } else {
            std::cerr << "TacxFTMS::update: received invalid no. of bytes: recv="
                      << status << ", got=" << status << std::endl;
        }
    }
    if (udpAlpine) {
        char tmpBuf[10000];
        int status;
        status = udpAlpine->receive(&tmpBuf, 10000);

        if (status == -1) {
            if (isEnabled())  // otherwise we are not supposed to receive
                              // anything
            {
                std::cerr << "Alpine::update: error while reading Alpine data"
                          << std::endl;
                if (udpAlpine)  // try to wake up the trainer (if the first start UDP
                          // message was lost)
                    udpAlpine->send("start");
            }
        } else if (status >= sizeof(AlpineData)) {
            if (!isEnabled()) {
                if (udpAlpine)  // still receiving data, send stop
                    udpAlpine->send("stop");
            }
            OpenThreads::ScopedLock<OpenThreads::Mutex> lock(mutex);
            memcpy(&alpineData, tmpBuf, sizeof(AlpineData));

        } else {
            std::cerr << "Alpine::update: received invalid no. of bytes: recv="
                      << status << ", got=" << status << std::endl;
        }
    } 
    else {
        usleep(5);
    }
    return;
}

void TacxFTMS::Initialize() {
    if (coVRMSController::instance()->isMaster()) {

        /*fluxControl.cmd = 1;

        ret = udpNeo->send(&fluxControl, sizeof(FluxCtrlData));*/
    }
}

float TacxFTMS::getSpeed() {
    return ftmsData.speed / 3.6;  // speed in m/s
}

float TacxFTMS::getAngle() { return alpineData.steering_angle; }

float TacxFTMS::getBrakeForce() {
    // prevents brake force to be negative
    /*if (ftmsData.brake < 0) {
        return 0;
    }*/
    return 0;
}

float TacxFTMS::getAccelleration() {
    // prevents brake force to be negative
    /*if (fluxData.brake < 0) {
        return 0;
    }
    return (fluxData.brake / 4.0) * 0.01;*/
    return 0.0;
}


void TacxFTMS::sendIndoorBikeSimulationParameters() {
    if (!udpNeo)
        return;

    udpNeo->send(&ftmsControl, sizeof(ftmsControl));
}

//void TacxFTMS::setResistance(float f) { resistance = f * 10.0; }

//void TacxFTMS::sendResistance() { ret = udp->send(&resistance, sizeof(float)); }

osgUtil::LineSegmentIntersector::Intersection getFirstIntersection(
    osg::ref_ptr<osg::LineSegment> ray, bool* haveISect) {
    //  just adjust height here

    osg::ref_ptr<osgUtil::IntersectorGroup> igroup =
        new osgUtil::IntersectorGroup;
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector;
    intersector =
        coIntersection::instance()->newIntersector(ray->start(), ray->end());
    igroup->addIntersector(intersector);

    osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Walk);
    VRSceneGraph::instance()->getTransform()->accept(visitor);

    *haveISect = intersector->containsIntersections();
    if (!*haveISect) {
        return {};
    }

    return intersector->getFirstIntersection();
}

osg::Matrix TacxFTMS::getMatrix() {
    float wheelDis = wheelBase * 1000;
    osg::Vec3 pos = TacxFTMSPos.getTrans();
    osg::Vec3d y{TacxFTMSPos(1, 0), TacxFTMSPos(1, 1), TacxFTMSPos(1, 2)};
    osg::Vec3 rearPos = pos + y * -wheelDis;

    osg::ref_ptr<osg::LineSegment> rayFront;
    {
        // down segment
        osg::Vec3 p0, q0;
        p0.set(pos[0], pos[1], pos[2] + stepSizeUp);
        q0.set(pos[0], pos[1], pos[2] - stepSizeDown);

        rayFront = new osg::LineSegment(p0, q0);
    }
    osg::ref_ptr<osg::LineSegment> rayBack;
    {
        // down segment
        osg::Vec3 p0, q0;

        p0.set(rearPos[0], rearPos[1], rearPos[2] + stepSizeUp);
        q0.set(rearPos[0], rearPos[1], rearPos[2] - stepSizeDown);

        rayBack = new osg::LineSegment(p0, q0);
    }
    bool intersects;
    auto front = getFirstIntersection(rayFront, &intersects);
    if (!intersects) {
        return TransformMat;
    }
    auto back = getFirstIntersection(rayBack, &intersects);
    if (!intersects) {
        return TransformMat;
    }

    auto frontNormal = front.getWorldIntersectNormal();
    frontNormal.normalize();
    auto backNormal = back.getWorldIntersectNormal();
    backNormal.normalize();
    // fprintf(stderr,"f %f \n",frontNormal*osg::Vec3(0,0,1));
    // fprintf(stderr,"b %f \n",backNormal*osg::Vec3(0,0,1) );
    if (frontNormal * osg::Vec3(0, 0, 1) < 0.2) return TransformMat;
    if (backNormal * osg::Vec3(0, 0, 1) < 0.2) return TransformMat;

    osg::Vec3d newY =
        front.getWorldIntersectPoint() - back.getWorldIntersectPoint();
    newY.normalize();
    //osg::Vec3d newX = newY ^ frontNormal;
    osg::Vec3d newX = newY ^ osg::Vec3(0,0,1);
    newX.normalize();
    osg::Vec3d newZ = newX ^ newY;

    osg::Vec3d translation = front.getWorldIntersectPoint();

    osg::Matrix newMatrix;

    newMatrix(0, 0) = newX.x();
    newMatrix(0, 1) = newX.y();
    newMatrix(0, 2) = newX.z();

    newMatrix(1, 0) = newY.x();
    newMatrix(1, 1) = newY.y();
    newMatrix(1, 2) = newY.z();

    newMatrix(2, 0) = newZ.x();
    newMatrix(2, 1) = newZ.y();
    newMatrix(2, 2) = newZ.z();

    newMatrix = newMatrix * osg::Matrix::translate(translation);

    osg::Matrix Nn = newMatrix;
    osg::Matrix invNn;
    invNn.invert(Nn);

    osg::Matrix NewTransform = TransformMat * invNn * TacxFTMSPos;
    osg::Vec3d z{NewTransform(2, 0), NewTransform(2, 1), NewTransform(2, 2)};

    if (z * osg::Vec3(0, 0, 1) < 0.2) return TransformMat;

    return NewTransform;
}

COVERPLUGIN(TacxFTMS)
