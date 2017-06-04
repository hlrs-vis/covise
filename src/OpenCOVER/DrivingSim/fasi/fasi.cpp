/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <fasi.h>
#include <unistd.h>
#include <net/covise_host.h>
#include <net/covise_socket.h>
#include <xenomai/init.h>

int main(int argc, char* const* argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: fasi file.xodr\n");
        return -1;
    }
    xenomai_init(&argc, &argv); 
    fasi *f = new fasi(argv[1]);
    f->run();
    delete f;

    return 1;
}

fasi *fasi::myFasi = NULL;

fasi::fasi(const char *filename)
{
    system = NULL;
    myFasi = this;
    serverConn = new covise::ServerConnection(31880, 1234, -1);
    if (!serverConn->getSocket())
    {
        std::cout << "tried to open server Port " << 31880 << std::endl;
        std::cout << "Creation of server failed!" << std::endl;
        std::cout << "Port-Binding failed! Port already bound?" << std::endl;
        exit(-1);
    }
    struct linger linger;
    linger.l_onoff = 0;
    linger.l_linger = 0;
    std::cout << "Set socket options..." << std::endl;
    if (serverConn)
    {
        setsockopt(serverConn->get_id(NULL), SOL_SOCKET, SO_LINGER, (char *)&linger, sizeof(linger));

        std::cout << "Set server to listen mode..." << std::endl;
        serverConn->listen();
        if (!serverConn->is_connected()) // could not open server port
        {
            fprintf(stderr, "Could not open server port %d\n", 31880);
            exit(-1);
        }
    }
    loadRoadSystem(filename);
    fum = fasiUpdateManager::instance();
    p_kombi = KI::instance();
    p_klsm = KLSM::instance();
    p_klima = Klima::instance();
    p_beckhoff = Beckhoff::instance();
    //p_brakepedal = BrakePedal::instance();
    p_gaspedal = GasPedal::instance();
    p_ignitionLock = IgnitionLock::instance();
    vehicleUtil = VehicleUtil::instance();

    sharedState.pedalA = 0; // Acceleration	[0,1]
    sharedState.pedalB = 0; // Brake				[0,1]
    sharedState.pedalC = 0; // Clutch			[0,1]

    sharedState.steeringWheelAngle = 0; // Wheel angle in Radians
    //sharedState.resetButton = false;			// Reset button	[true, false]
    sharedState.gear = 0; // Present gear	[-1, 0, 1, ...]
    //sharedState.hornButton = false;			// Horn button		[true, false]

    oldFanButtonState = false;
    oldParkState = true;
    automatic = true;

    vehicleDynamics = new FourWheelDynamicsRealtime();

    fprintf(stderr, "\n\n\ninit KI\n");

    p_beckhoff->setDigitalOut(0, 0, false);
    p_beckhoff->setDigitalOut(0, 1, false);
    p_beckhoff->setDigitalOut(0, 2, false);
    p_beckhoff->setDigitalOut(0, 3, false);
    p_beckhoff->setDigitalOut(0, 4, false);
    p_beckhoff->setDigitalOut(0, 5, false);
    p_beckhoff->setDigitalOut(0, 6, false);
    p_beckhoff->setDigitalOut(0, 7, false);
}

fasi::~fasi()
{
    vehicleDynamics->platformToGround();
    delete vehicleDynamics;
    delete fum;
}

bool fasi::loadRoadSystem(const char *filename_chars)
{
    std::string filename(filename_chars);
    std::cerr << "Loading road system!" << std::endl;
    if (system == NULL)
    {
        //Building directory string to xodr file
        xodrDirectory.clear();
        if (filename[0] != '/' && filename[0] != '\\' && (!(filename[1] == ':' && (filename[2] == '/' || filename[2] == '\\'))))
        { // / or backslash or c:/
            char *workingDir = getcwd(NULL, 0);
            xodrDirectory.assign(workingDir);
            free(workingDir);
        }
        size_t lastSlashPos = filename.find_last_of('/');
        size_t lastSlashPos2 = filename.find_last_of('\\');
        if (lastSlashPos != filename.npos && (lastSlashPos2 == filename.npos || lastSlashPos2 < lastSlashPos))
        {
            if (!xodrDirectory.empty())
                xodrDirectory += "/";
            xodrDirectory.append(filename, 0, lastSlashPos);
        }
        if (lastSlashPos2 != filename.npos && (lastSlashPos == filename.npos || lastSlashPos < lastSlashPos2))
        {
            if (!xodrDirectory.empty())
                xodrDirectory += "\\";
            xodrDirectory.append(filename, 0, lastSlashPos2);
        }

        system = RoadSystem::Instance();

        xercesc::DOMElement *openDriveElement = getOpenDriveRootElement(filename);
        if (!openDriveElement)
        {
            std::cerr << "No regular xodr file " << filename << " at: " + xodrDirectory << std::endl;
            return false;
        }

        system->parseOpenDrive(openDriveElement);
        this->parseOpenDrive(rootElement);
    }
    return true;
}

xercesc::DOMElement *fasi::getOpenDriveRootElement(std::string filename)
{
    try
    {
        xercesc::XMLPlatformUtils::Initialize();
    }
    catch (const xercesc::XMLException &toCatch)
    {
        char *message = xercesc::XMLString::transcode(toCatch.getMessage());
        std::cout << "Error during initialization! :\n" << message << std::endl;
        xercesc::XMLString::release(&message);
        return NULL;
    }

    xercesc::XercesDOMParser *parser = new xercesc::XercesDOMParser();
    parser->setValidationScheme(xercesc::XercesDOMParser::Val_Never);

    try
    {
        parser->parse(filename.c_str());
    }
    catch (...)
    {
        std::cerr << "Couldn't parse OpenDRIVE XML-file " << filename << "!" << std::endl;
    }

    xercesc::DOMDocument *xmlDoc = parser->getDocument();
    if (xmlDoc)
    {
        rootElement = xmlDoc->getDocumentElement();
    }

    return rootElement;
}

void fasi::parseOpenDrive(xercesc::DOMElement *rootElement)
{
    xercesc::DOMNodeList *documentChildrenList = rootElement->getChildNodes();

    for (int childIndex = 0; childIndex < documentChildrenList->getLength(); ++childIndex)
    {
        xercesc::DOMElement *sceneryElement = dynamic_cast<xercesc::DOMElement *>(documentChildrenList->item(childIndex));
        if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), xercesc::XMLString::transcode("scenery")) == 0)
        {
            /*

   std::string fileString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("file")));
   std::string vpbString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("vpb")));

   std::vector<BoundingArea> voidBoundingAreaVector;
   std::vector<std::string> shapeFileNameVector;

   xercesc::DOMNodeList* sceneryChildrenList = sceneryElement->getChildNodes();
   xercesc::DOMElement* sceneryChildElement;
   for(unsigned int childIndex=0; childIndex<sceneryChildrenList->getLength(); ++childIndex) {
sceneryChildElement = dynamic_cast<xercesc::DOMElement*>(sceneryChildrenList->item(childIndex));
if(!sceneryChildElement) continue;

if(xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), xercesc::XMLString::transcode("void"))==0) {
double xMin = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("xMin"))));
double yMin = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("yMin"))));
double xMax = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("xMax"))));
double yMax = atof(xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("yMax"))));

voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(xMin, yMin),osg::Vec2(xMax, yMax)));
//voidBoundingAreaVector.push_back(BoundingArea(osg::Vec2(506426.839,5398055.357),osg::Vec2(508461.865,5399852.0)));
}
else if(xercesc::XMLString::compareIString(sceneryChildElement->getTagName(), xercesc::XMLString::transcode("shape"))==0) {
std::string fileString = xercesc::XMLString::transcode(sceneryChildElement->getAttribute(xercesc::XMLString::transcode("file")));
shapeFileNameVector.push_back(fileString);
}

}
*/

            /*
         if(!fileString.empty())
         {
            if(!coVRFileManager::instance()->fileExist((xodrDirectory+"/"+fileString).c_str()))
            {
               std::cerr << "\n#\n# file not found: this may lead to a crash! \n#" << endl;
            }
            coVRFileManager::instance()->loadFile((xodrDirectory+"/"+fileString).c_str());
         }

         if(!vpbString.empty())
{
coVRPlugin* roadTerrainPlugin = cover->addPlugin("RoadTerrain");
fprintf(stderr,"loading %s\n",vpbString.c_str());
if(RoadTerrainPlugin::plugin)
{
osg::Vec3d offset(0,0,0);
const RoadSystemHeader& header = RoadSystem::Instance()->getHeader();
offset.set(header.xoffset, header.yoffset, 0.0);
fprintf(stderr,"loading %s offset: %f %f\n",(xodrDirectory+"/"+vpbString).c_str(),offset[0],offset[1]);
RoadTerrainPlugin::plugin->loadTerrain(xodrDirectory+"/"+vpbString,offset, voidBoundingAreaVector, shapeFileNameVector);
}
}*/
        }
        else if (sceneryElement && xercesc::XMLString::compareIString(sceneryElement->getTagName(), xercesc::XMLString::transcode("environment")) == 0)
        {
            std::string startRoadString = xercesc::XMLString::transcode(sceneryElement->getAttribute(xercesc::XMLString::transcode("startRoad")));
            if(startRoadString.length()>0)
            {
            }
        }
    }
}

int fasi::getAutoGearDiff(float downRPM, float upRPM)
{
    int diff = 0;

    double speed = vehicleDynamics->getEngineSpeed();

    if (sharedState.gear == 0)
    {
        if (speed < 10)
            diff = -1;
        else if (speed > 20)
            diff = 1;
    }
    else if (sharedState.gear == 1)
    {
        if (speed < 10)
            diff = -1;
        else if (speed > upRPM)
            diff = 1;
    }
    else
    {
        if (speed < downRPM)
            diff = -1;
        else if (speed > upRPM)
            diff = 1;
    }

    return diff;
}

bool fasi::readClientVal(void *buf, unsigned int numBytes)
{
    unsigned int toRead = numBytes;
    unsigned int numRead = 0;
    int readBytes = 0;
    if (toClientConn == NULL)
        return false;
    while (numRead < numBytes)
    {
        readBytes = toClientConn->getSocket()->Read(((unsigned char *)buf) + readBytes, toRead);
        if (readBytes < 0)
        {
            std::cout << "error reading data from socket" << std::endl;
            delete toClientConn;
            toClientConn = NULL;
            return false;
        }
        numRead += readBytes;
        toRead = numBytes - numRead;
    }
    return true;
}

void fasi::run()
{
    bool running = true;
    bool wasRunning = false;
    vehicleDynamics->platformToGround();
    toClientConn = NULL;
    while (running)
    {
        fum->update();

        // we have a server and received a connect
        if ((!toClientConn) && serverConn && serverConn->is_connected() && serverConn->check_for_input())
        {
            //   std::cout << "Trying serverConn..." << std::endl;
            toClientConn = serverConn->spawnSimpleConnection();
            if (toClientConn && toClientConn->is_connected())
            {
            }
            else
            {
                toClientConn = NULL;
            }
        }
        if (toClientConn)
        {
            if (!readClientVal(&sharedState.frameTime, sizeof(sharedState.frameTime)))
            {
                std::cout << "Creset " << sharedState.frameTime << std::endl;
            }

            sharedState.steeringWheelAngle = vehicleDynamics->getSteerWheelAngle();
            vehicleDynamics->move();
            remoteData.V = vehicleDynamics->getVelocity();
            remoteData.A = vehicleDynamics->getAcceleration();
            remoteData.rpm = vehicleDynamics->getEngineSpeed();
            remoteData.torque = vehicleDynamics->getEngineTorque();
            remoteData.chassisTransform = vehicleDynamics->getVehicleTransformation();
            remoteData.gear = sharedState.gear;
            remoteData.buttonStates = 0;
            remoteData.buttonStates |= p_kombi->getLightState() << 8;
            remoteData.buttonStates |= p_kombi->getJoystickState() << 16;
            if ((bool)(p_klsm->getHornStat()))
                remoteData.buttonStates |= 1;
            if ((bool)(p_klsm->getReturnStat()))
                remoteData.buttonStates |= 2;
            if ((bool)(p_klsm->getBlinkLeftStat()))
                remoteData.buttonStates |= 4;
            if ((bool)(p_klsm->getBlinkRightStat()))
                remoteData.buttonStates |= 8;
            if (p_ignitionLock->getLockState() == IgnitionLock::ENGINESTART)
                remoteData.buttonStates |= 16;
            if (toClientConn)
            {
                int written = toClientConn->getSocket()->write(&remoteData, sizeof(remoteData));
                if (written < 0)
                {
                    delete toClientConn;
                    toClientConn = NULL;
                    std::cout << "Cresetw " << sharedState.frameTime << std::endl;
                }
            }
        }
        else
        {
            usleep(100000);
            if (vehicleDynamics)
            {
                sharedState.steeringWheelAngle = vehicleDynamics->getSteerWheelAngle();
                vehicleDynamics->move(); // move car forward
            }
            timeval currentTime;
            gettimeofday(&currentTime, NULL);
            sharedState.frameTime = (currentTime.tv_sec + (double)currentTime.tv_usec / 1000000.0);
        }

        /* if(vehicleUtil->getVehicleState()==VehicleUtil::KEYIN_ERUNNING)
        {
            wasRunning =true;
        }
        else
        {
            if(wasRunning)
                running = false;
        }*/
        if (sharedState.PSMState)
        {
            running = false; // Stop fasi
        }

        float ccGas = 0.0;

        p_klsm->p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_GRA_D = 1;
        /*if(ccOn)
{
if(ccActive)
{
   float sDiff = ccSpeed - SteeringWheelPlugin::plugin->dynamics->getVelocity();
   sDiff = sDiff/10.0;
   if(sDiff > 0.8)
       sDiff = 0.8;
   iDiff+=sDiff*0.01*cover->frameDuration();
   ccGas = iDiff + sDiff;
   if(ccGas > 0.8)
ccGas = 0.8;

}
if(ccGas < 0)
ccGas=0;
}
else
{
iDiff=0;
p_klsm->p_CANProv->GW_SVB_D.values.canmsg.cansignals.SVB_GRA_D = 0;

}*/
        bool ParkState = false;

        static bool firstTime = true;
        if (firstTime)
        {

            if (vehicleDynamics)
            {
                vehicleDynamics->platformToGround();
            }

            firstTime = false;
        }

        if (vehicleUtil->getVehicleState() == VehicleUtil::KEYIN_ERUNNING)
        {
            float pedal = p_gaspedal->getActualAngle() / 100.0;
            if (pedal > 0.01)
            {
                //fprintf(stderr, "gas %f\n", pedal);
            }
            if (ccGas > pedal)
                sharedState.pedalA = ccGas;
            else
                sharedState.pedalA = pedal;
        }
        else
        {
            ParkState = true;
        }
        sharedState.pedalB = std::max(0.0, std::min(1.0, ValidateMotionPlatform::instance()->getBrakePedalPosition() * 5.0));
        sharedState.pedalC = 0;

        //resetButton = p_klsm->getReturnStat();

        p_kombi->setPetrolLevel(100);
        static int oldState = -1;
        static bool startEngine = false;
        //static double startTime = 0.0;

        int currentLeverState = 0;
        if (p_beckhoff->getDigitalIn(0, 0))
        {
            if (p_beckhoff->getDigitalIn(0, 1))
            {
                if (p_beckhoff->getDigitalIn(0, 7))
                {
                    automatic = false;
                    //M
                    p_kombi->setGearshiftLever(KI::GEAR_M);
                    currentLeverState = KI::GEAR_M;
                }
                else if (!p_beckhoff->getDigitalIn(0, 2))
                {
                    automatic = true;
                    //D
                    p_kombi->setGearshiftLever(KI::GEAR_D);
                    currentLeverState = KI::GEAR_D;
                }
            }
            else
            {
                sharedState.gear = -1;
                automatic = false;
                //R
                p_kombi->setGearshiftLever(KI::GEAR_R);
                currentLeverState = KI::GEAR_R;
            }
        }
        else if (p_beckhoff->getDigitalIn(0, 1))
        {
            sharedState.gear = 0;
            ParkState = true;
            //P
            p_kombi->setGearshiftLever(KI::GEAR_P);
            currentLeverState = KI::GEAR_P;
            automatic = false;
        }
        else if (p_beckhoff->getDigitalIn(0, 2))
        {
            automatic = false;
            //N
            p_kombi->setGearshiftLever(KI::GEAR_N);
            currentLeverState = KI::GEAR_N;
            sharedState.gear = 0;
        }

        if (oldState != vehicleUtil->getVehicleState())
        {
            std::cerr << "currentState:" << vehicleUtil->getVehicleState() << std::endl;
            if (vehicleUtil->getVehicleState() == VehicleUtil::KEYIN)
                p_ignitionLock->releaseKey();
        }
        oldState = vehicleUtil->getVehicleState();
        if (p_ignitionLock->getLockState() == IgnitionLock::KEYOUT)
        {
            if (vehicleUtil->getVehicleState() != VehicleUtil::KEYOUT)
            {
                std::cerr << "out" << std::endl;
                vehicleUtil->setVehicleState(VehicleUtil::KEYOUT);
            }
        }
        else if (p_ignitionLock->getLockState() == IgnitionLock::ENGINESTOP)
        {
            // key_left (STOP ENGINE)
            std::cerr << "stop" << std::endl;
            vehicleUtil->setVehicleState(VehicleUtil::KEYIN_ESTOP);
        }
        else if (p_ignitionLock->getLockState() == IgnitionLock::IGNITION)
        {
            // key_right1 (IGNITION)
            if (vehicleUtil->getVehicleState() == VehicleUtil::KEYIN)
            {
                std::cerr << "ignite" << std::endl;
                vehicleUtil->setVehicleState(VehicleUtil::KEYIN_IGNITED);
            }
        }
        else if (p_ignitionLock->getLockState() == IgnitionLock::ENGINESTART && currentLeverState == KI::GEAR_P)
        {
            // key_right2 (START ENGINE)
            if (startEngine == false && vehicleUtil->getVehicleState() == VehicleUtil::KEYIN_IGNITED)
            {
                startEngine = true;
                //startTime = cover->frameTime();
                /*#ifdef USE_CAR_SOUND
        CarSound::instance()->start(CarSound::Ignition);
#else
             anlasserSource->play();
#endif*/
                std::cerr << "start" << vehicleUtil->getVehicleState() << "  " << VehicleUtil::KEYIN_IGNITED << " " << startEngine << std::endl;
                vehicleUtil->setVehicleState(VehicleUtil::KEYIN_ESTART);
            }
        }
        else if (p_ignitionLock->getLockState() == IgnitionLock::KEYIN)
        {
            if (vehicleUtil->getVehicleState() == VehicleUtil::KEYOUT)
            {
                std::cerr << "in" << std::endl;
                vehicleUtil->setVehicleState(VehicleUtil::KEYIN);
            }
        }
        if (startEngine && vehicleUtil->getVehicleState() == VehicleUtil::KEYIN_ESTART /*&& cover->frameTime()> startTime +0.6*/)
        {
            vehicleUtil->setVehicleState(VehicleUtil::KEYIN_ERUNNING);
            startEngine = false;
        }

        if (ParkState && sharedState.pedalB > 0.1 && vehicleUtil->getVehicleState() == VehicleUtil::KEYIN_ERUNNING)
        {
            p_beckhoff->setDigitalOut(0, 0, true);
        }
        else
        {
            p_beckhoff->setDigitalOut(0, 0, false);
        }
        if (p_klima->getFanButtonStat() && oldFanButtonState == false)
        {
            if (vehicleDynamics)
            {
                vehicleDynamics->centerWheel();
            }
            /*#ifdef HAVE_CARDYNAMICSCGA
  else if(cardynCGA) {
      cardynCGA->centerSteeringWheel();
  }
#endif*/
        }
        oldFanButtonState = p_klima->getFanButtonStat();
        if (ParkState && !oldParkState)
        {

            if (vehicleDynamics)
            {
                vehicleDynamics->platformToGround();
            }
        }
        if (!ParkState && oldParkState)
        {

            if (vehicleDynamics)
            {
                vehicleDynamics->platformReturnToAction();
            }
        }

        oldParkState = ParkState;

        static unsigned char oldButtonState = 0;

        if (p_kombi->getButtonState() & KI::Button_Sport && !(oldButtonState & KI::Button_Sport))
        {
            sharedState.SportMode = (!sharedState.SportMode);
        }
        if (p_kombi->getButtonState() & KI::Button_PSM && !(oldButtonState & KI::Button_PSM))
        {
            sharedState.PSMState = (!sharedState.PSMState);
        }
        if (p_kombi->getButtonState() & KI::Button_Spoiler && !(oldButtonState & KI::Button_Spoiler))
        {
            sharedState.SpoilerState = (!sharedState.SpoilerState);
        }
        if (p_kombi->getButtonState() & KI::Button_Damper && !(oldButtonState & KI::Button_Damper))
        {
            sharedState.DamperState = (!sharedState.DamperState);
            if (vehicleDynamics)
            {
                vehicleDynamics->setSportDamper(sharedState.DamperState);
            }
        }
        oldButtonState = p_kombi->getButtonState();
        unsigned char leds = 0;
        leds |= sharedState.SpoilerState << 0;
        leds |= sharedState.DamperState << 1;
        leds |= sharedState.SportMode << 2;
        leds |= sharedState.PSMState << 3;
        //fprintf(stderr,"leds %d\n" ,leds);
        p_kombi->setLEDState(leds);

        //std::cerr << "p_beckhoff digitil in 0:" << (int)p_beckhoff->getDigitalIn(0) << ", 1: "  << (int)p_beckhoff->getDigitalIn(1) << ", 2: "  << (int)p_beckhoff->getDigitalIn(2) << std::endl;
        //fprintf(stderr,"automatic %d  %lf \n", automatic,sharedState.frameTime);
        if (automatic)
        {
            static double oldShiftTime = 0;
            if (sharedState.frameTime - oldShiftTime > 0.2)
            {
                int gearDiff;
                if (sharedState.SportMode)
                {
                    gearDiff = getAutoGearDiff(55, 108);
                }
                else
                {
                    //fprintf(stderr,"gas %f %f \n" ,sharedState.pedalA,SteeringWheelPlugin::plugin->dynamics->getEngineSpeed());

                    gearDiff = getAutoGearDiff(19.0 + (35 * sharedState.pedalA * sharedState.pedalA), 32 + (73 * sharedState.pedalA * sharedState.pedalA));
                }
                sharedState.gear += gearDiff;
                if (sharedState.gear < 0)
                {
                    sharedState.gear = 0;
                }
                else if (sharedState.gear > 5)
                {
                    sharedState.gear = 5;
                }
                fprintf(stderr,"geardiff %d %d\n",gearDiff,sharedState.gear);
                if (gearDiff != 0)
                {
                    oldShiftTime = sharedState.frameTime;
                }
            }
        }
        else
        {
            static int oldStat = 0;
            int stat = p_klsm->getShiftStat();
            if (stat != oldStat)
            {
                sharedState.gear += stat;
                if (sharedState.gear < 0)
                {
                    sharedState.gear = 0;
                }
                else if (sharedState.gear > 5)
                {
                    sharedState.gear = 5;
                }
            }
            oldStat = stat;
        }
        if (p_klsm->getBlinkLeftStat())
        {
            p_kombi->indicator(BlinkerTask::LEFT);
        }
        else if (p_klsm->getBlinkRightStat())
        {
            p_kombi->indicator(BlinkerTask::RIGHT);
        }
        else
        {
            p_kombi->indicator(BlinkerTask::NONE);
        }
        p_kombi->setGear(sharedState.gear);

        if (vehicleUtil->getVehicleState() == VehicleUtil::KEYIN_ERUNNING)
        {
            p_kombi->setSpeed(vehicleDynamics->getVelocity() * 3.6);
            p_kombi->setRevs(vehicleDynamics->getEngineSpeed() * 60);
        }
        else
        {
            p_kombi->setSpeed(0.0);
            p_kombi->setRevs(0.0);
        }
        if ((bool)(p_klsm->getReturnStat()))
        {
            if (vehicleDynamics)
            {
                vehicleDynamics->resetState();
            }
        }

        //hornButton = (bool)(p_klsm->getHornStat());
    }
}
