/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PorscheRealtimeDynamics.h"


#include <OpenVRUI/osg/mathUtils.h>
#include <config/CoviseConfig.h>
#include "SteeringWheel.h"

#include <cover/coVRTui.h>
#include <cover/coIntersection.h>

#include <net/covise_host.h>
#include <net/covise_socket.h>

#include <string>
#include <math.h>

#include <fstream>

using covise::coCoviseConfig;

double PorscheRealtimeDynamics::dSpace_v = -1;

extern float rayHeight = -1.0f;
// CONSTRUCTOR //
//
PorscheRealtimeDynamics::PorscheRealtimeDynamics()
{

    // config XML //
    serverPort = coCoviseConfig::getInt("port", "COVER.Plugin.SteeringWheel.Dynamics.PorscheRealtimeServer", 52001);
    localPort = coCoviseConfig::getInt("localPort", "COVER.Plugin.SteeringWheel.Dynamics.PorscheRealtimeServer", 52001);
    std::string remoteHost = coCoviseConfig::getEntry("host", "COVER.Plugin.SteeringWheel.Dynamics.PorscheRealtimeServer");
    filestring = coCoviseConfig::getEntry("file", "COVER.Plugin.SteeringWheel.Dynamics.PorscheRealtimeServer", "dynamics.txt");
    serverHost = NULL;

    doRun = false;
    if (!remoteHost.empty())
    {
        serverHost = new Host(remoteHost.c_str());
    }

    oldTime = 0;
    oldHeight = 0;

    chassisTransform.makeIdentity();
    ffz1chassisTransform.makeIdentity();

    dSpaceReference.makeIdentity();
    dSpaceOrigin.makeIdentity();
    dSpacePose.makeIdentity();

    outputData[0] = 0;

    toDSPACE = NULL;
    if (coVRMSController::instance()->isMaster())
    {
        if (!remoteHost.empty() && coVRMSController::instance()->isMaster())
        {
            toDSPACE = new UDPComm(remoteHost.c_str(), serverPort, localPort);
            doRun = true;
            startThread();
            cerr << "starting thread" << endl;
        }
    }
    // Robert - Initialisierung des oldnormal-Vektors
    initOldnormal();
    oldarr[0] = 0.0;
    oldarr[1] = 0.0;
    oldarr[2] = 0.0;
    oldarr[3] = 0.0;
    oldarr[4] = 0.0;
    oldarr[5] = 0.0;

    //NEU 16-02-2011
    // oX = 0.0;
    //	oY = 0.0;
}

// DESTRUCTOR //
//
PorscheRealtimeDynamics::~PorscheRealtimeDynamics()
{
    doRun = false;
    if (coVRMSController::instance()->isMaster())
    {
        fprintf(stderr, "waiting1\n");
        endBarrier.block(2); // wait until communication thread finishes
        fprintf(stderr, "done1\n");
    }
    //inStream.close();
}

void
PorscheRealtimeDynamics::update()
{
    if (coVRMSController::instance()->isMaster())
    {
        memcpy((void *)&DSpaceData, (void *)&threadDSpaceData, sizeof(DSpaceData));
        coVRMSController::instance()->sendSlaves((char *)&DSpaceData, sizeof(DSpaceData));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&DSpaceData, sizeof(DSpaceData));
    }
}

void
PorscheRealtimeDynamics::run()
{
    // receiving and sending thread, also does the low level simulation like hard limits

    memset(&threadDSpaceData, 0, sizeof(threadDSpaceData));

    while (doRun)
    {
        sendData();
        while (doRun && !readData())
        {
            microSleep(10);
        }
    }
    fprintf(stderr, "waiting2\n");
    endBarrier.block(2);
    fprintf(stderr, "done2\n");
}

// SEND //
//
// inactive
bool
PorscheRealtimeDynamics::sendData()
{
    //toDSPACE->send(&sendBuffer,sizeof(sendBuffer));
    //sendBuffer.reset = 0;
    return true;
}

// READ //
//
// dSPACE to VIS
bool
PorscheRealtimeDynamics::readData()
{
    int ret = toDSPACE->readMessage();
    if (ret < 4)
        return false;
    int *msgType = (int *)toDSPACE->rawBuffer();
    switch (*msgType)
    {
    case DSpaceToVis:
        if (toDSPACE->messageSize() == sizeof(MSG_DSpaceToVis))
        {
            memcpy((void *)&threadDSpaceData, toDSPACE->rawBuffer(), sizeof(MSG_DSpaceToVis));
            // Testausgabe von msgBuf (14.04.2011)
            if (coVRMSController::instance()->isMaster())
            {
                //ofstream file;
                //file.open("rawBuf.txt", ios::out | ios::app);
                //file << "msgBuf: "<< toDSPACE->rawBuffer() << endl;
                //file.close();
            }
        }
        else
        {
            cerr << "received wrong UDP Message, expected type " << DSpaceToVis << " size " << sizeof(MSG_DSpaceToVis) << endl << "but received type " << *msgType << " size " << toDSPACE->messageSize() << endl;
        }
        break;
    default:
        cerr << "received unknown message type " << *msgType << " size " << toDSPACE->messageSize() << endl;
        break;
    }
    return true;
} // returns true on success, false if no data has been received.

bool
PorscheRealtimeDynamics::readFile(double *data, unsigned int number)
{
    if (inStream.eof())
    {
        inStream.close();
        inStream.open(filestring.c_str(), std::ios::in);
        std::string headerString;
        getline(inStream, headerString);
        std::cerr << "Reading input file again, dynamics.txt header: " << headerString << std::endl;
    }
    else
    {
        unsigned int i = 0;
        while (i < number)
        {
            inStream >> data[i];
            ++i;
        }
    }

    return true;
}

// place vehicle on ground
void
PorscheRealtimeDynamics::moveToStreet(osg::Matrix &carTrans, osg::Matrix &moveMat)
{

    osg::Matrix carTransWorld, tmp;
    osg::Vec3 pos, p0, q0;

    osg::Matrix baseMat;
    osg::Matrix invBaseMat;

    baseMat = cover->getObjectsScale()->getMatrix();

    osg::Matrix transformMatrix = cover->getObjectsXform()->getMatrix();

    baseMat.postMult(transformMatrix);
    invBaseMat.invert(baseMat);

    osg::Matrix invVRMLRotMat;
    invVRMLRotMat.makeRotate(-M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
    osg::Matrix VRMLRotMat;
    VRMLRotMat.makeRotate(M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));

    //Testausgabe fuer Koordinaten von carTrans
    // cout << "carTrans: x= " << carTrans.getTrans()[0] << " y= " << carTrans.getTrans()[1] << " z= " << carTrans.getTrans()[2];
    carTransWorld = carTrans * VRMLRotMat * baseMat;
    pos = carTransWorld.getTrans();
    pos[2] += oldHeight;

    if (false)
    { /*

   //Auswertung von 25 Strahlen:
   //Koordinaten einlsen
   //float coordA[50];
   float height_ds = DSpaceData.PosZ;
   osg::Vec3 s0,t0;
   //coordA = DSpaceData.coordA;

   // Strahlen generieren
   osg::ref_ptr<osg::LineSegment> rayA[25]= new osg::LineSegment();
   int a = 0;
   for (int i=0; i<25; i++)
   {
      s0.set(-DSpaceData.coordA[a+1],  DSpaceData.coordA[a], height_ds+40000.0); // Strahlen +- 40 m
      t0.set(-DSpaceData.coordA[a+1],  DSpaceData.coordA[a], height_ds-40000.0);
      rayA[i]->set(s0, t0);
      a +=2;
   }

   // Schnittpunkte auslesen
   osgUtil::IntersectVisitor visitorR;
   visitorR.setTraversalMask(Isect::Collision);
   for (int w=0; w<25; w++)
   { //Strahlen der TraversalMask hinzufuegen, die den Szenegraphen durchlaeuft
      visitorR.addLineSegment(rayA[w].get());
   }

   cover->getObjectsXform()->accept(visitorR);

   int test[25]; // Falls SP vorhanden, wird in diese Variable geschrieben
   for (int nr=0; nr<25; nr++)
   { //Ueberpruefung ob es SP gab
      test[nr]= visitorR.getNumHits(rayA[nr].get());
      cout << ">>>>>>>>SP:" << test[nr] << endl;
   }

   osgUtil::Hit hitInformationA[25];
   float hoehe[25];
   for (int h=0; h<25; h++)
   {
      if (test[h])
      {
         hitInformationA[h] = visitorR.getHitList(rayA[h].get()).front();
         hoehe[h] = hitInformationA[h].getWorldIntersectPoint()[2];
         cout << hoehe[h] << " " << h << endl;
      }
   }
   int udpcheck = toDSPACE->send(&hoehe,sizeof(hoehe));
   cout << "udpcheck: " << udpcheck << endl;
*/
    }
    //////

    //////////////////////////////////
    // 		   TEST 20 Strahlen			  //
    //////////////////////////////////
    /*
   // Strahlen generieren
   osg::ref_ptr<osg::LineSegment> rayA[20]= new osg::LineSegment();;
   for (int i=0; i<20; i++) {
   //	rayA[i] = new osg::LineSegment();
      p0.set(pos[0]+2000+i*20,  pos[1]+1500,  pos[2]+1500.0);
      q0.set(pos[0]+2000+i*20,  pos[1]+1500, pos[2]-40000.0);
      rayA[i]->set(p0, q0);
   }

   // Schnittpunkte auslesen
   osgUtil::IntersectVisitor visitorS;
   visitorS.setTraversalMask(Isect::Collision);
   for (int w=0; w<20; w++) { //Strahlen der TraversalMask hinzufuegen, die den Szenegraphen durchlaeuft
      visitorS.addLineSegment(rayA[w].get());
   }

   cover->getObjectsXform()->accept(visitorS);

   int test[20]; // Falls SP vorhanden, wird in diese Variable geschrieben
   for (int nr=0; nr<20; nr++) { //Ueberpruefung ob es SP gab
   test[nr]= visitorS.getNumHits(rayA[nr].get());
   cout << ">>>>>>>>" << test[nr] << endl;
   }

   osgUtil::Hit hitInformationA[20];
   double hoehe[20];
   for (int h=0; h<20; h++) {
            if (test[h]) {
            hitInformationA[h] = visitorS.getHitList(rayA[h].get()).front();
            hoehe[h] = hitInformationA[h].getWorldIntersectPoint()[2];
            //cout << hitInformationA[h] << endl;
            cout << hoehe[h] << " " << h << endl;
         }
   }
*/
    /*
// KOMPAKTE VERSION
   osg::ref_ptr<osg::LineSegment> rayA[20];
   osgUtil::IntersectVisitor visitorS;
   visitorS.setTraversalMask(Isect::Collision);
   cover->getObjectsXform()->accept(visitorS);
   int test[20]; // Falls SP vorhanden, wird in diese Variable geschrieben
   osgUtil::Hit hitInformationA[20];
   double hoehe[20];

   for (int i=0; i<20; i++) {
      rayA[i] = new osg::LineSegment();
      p0.set(pos[0]+2000+i*20,  pos[1]+1500,  pos[2]+1500.0);
      q0.set(pos[0]+2000+i*20,  pos[1]+1500, pos[2]-40000.0);
      rayA[i]->set(p0, q0);

      visitorS.addLineSegment(rayA[i].get());//Strahlen der TraversalMask hinzufuegen, die den Szenegraphen durchlaeuft

      test[i]= visitorS.getNumHits(rayA[i].get());



      if (test[i]) {
            hitInformationA[i] = visitorS.getHitList(rayA[i].get()).front();
            hoehe[i] = hitInformationA[i].getWorldIntersectPoint()[2];
            cout << "Hoehe["<<i<<"]" <<hoehe[i] << sendl;
         }
   }
// ENDE KOMPAKT
*/
    //////////////////////////////////
    // 		   ENDE 20 Strahlen			  //
    //////////////////////////////////

    //////////////////////////////////
    // 		   Reference Points			  //
    //////////////////////////////////

    //----------------------//
    //  ... for plane 1     //
    //----------------------//

    osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector[6];

    // Front-Right segment
    p0.set(pos[0] + 2000, pos[1] + 1500, pos[2] + 1500.0); // 1.5 m above actual position and ...
    q0.set(pos[0] + 2000, pos[1] + 1500, pos[2] - 40000.0); // 40 m under actual position
    intersector[0] = coIntersection::instance()->newIntersector(p0,q0);
    igroup->addIntersector(intersector[0]);

    // Back-Center segment
    p0.set(pos[0], pos[1] - 1500, pos[2] + 1500.0);
    q0.set(pos[0], pos[1] - 1500, pos[2] - 40000.0);
    intersector[1] = coIntersection::instance()->newIntersector(p0, q0);
    igroup->addIntersector(intersector[1]);

    // Front-Left segment
    p0.set(pos[0] - 2000, pos[1] + 1500, pos[2] + 1500.0);
    q0.set(pos[0] - 2000, pos[1] + 1500, pos[2] - 40000.0);
    intersector[2] = coIntersection::instance()->newIntersector(p0, q0);
    igroup->addIntersector(intersector[2]);

    //----------------------//
    //  ... for plane 2     //
    //----------------------//

    // Front-center segment
    p0.set(pos[0], pos[1] + 1500, pos[2] + 1500.0);
    q0.set(pos[0], pos[1] + 1500, pos[2] - 40000.0);
    intersector[3] = coIntersection::instance()->newIntersector(p0, q0);
    igroup->addIntersector(intersector[3]);

    // Back-Right segment
    p0.set(pos[0] + 2000, pos[1] - 1500, pos[2] + 1500.0);
    q0.set(pos[0] + 2000, pos[1] - 1500, pos[2] - 40000.0);
    intersector[4] = coIntersection::instance()->newIntersector(p0, q0);
    igroup->addIntersector(intersector[4]);

    // Back-Left segment
    p0.set(pos[0] - 2000, pos[1] - 1500, pos[2] + 1500.0);
    q0.set(pos[0] - 2000, pos[1] - 1500, pos[2] - 40000.0);
    intersector[5] = coIntersection::instance()->newIntersector(p0, q0);
    igroup->addIntersector(intersector[5]);

    osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Collision);
    cover->getObjectsXform()->accept(visitor);


    ////////////////////////////////////////
    // 		   Intersection check		  //
    ////////////////////////////////////////

    osg::Vec3 point[6];
    int numHits = 0;
    for (int i = 0; i < 6; i++)
    {
        if (intersector[0]->containsIntersections())
        {
            point[i] = intersector[0]->getFirstIntersection().getWorldIntersectPoint();
            numHits++;
        }
    }

    if (numHits==6)
    {
        float dist = 0.0;
        osg::Vec3 normal(0, 0, 1);
        osg::Vec3 oldnormal(0, 0, 1);
        osg::Vec3 realnormal(0, 0, 1);
        osg::Geode *geode = NULL;

        /////////////////////////////////////////////////////////////////////////
        //          Creating a normal out of the normals of two planes         //
        /////////////////////////////////////////////////////////////////////////

        //------------------------------//
        //         create planes        //
        //------------------------------//

        osg::Plane plane(point[0], point[1], point[2]);
        osg::Plane plane2(point[3], point[4], point[5]);

        //------------------------------//
        //        create normals        //
        //------------------------------//

        normal = (plane.getNormal() + plane2.getNormal());
        normal.normalize();
        realnormal = normal;
        oldnormal = getOldnormal();

        //Vergleich mit altem Normalenvektor. Falls zu große Abweichung -> Einschränkung

        ////////////////////////////////////////////////////////////////
        //     Neuer Ansatz mit Beruecksichtigung der Vorzeichen:     //
        ////////////////////////////////////////////////////////////////

        if (oldnormal[0] != 0 || oldnormal[1] != 0 || oldnormal[2] != 0)
        { // 1. Durchlauf, initialisierung des oldnormal-Vektors

            float up = oldnormal[0] * 1.25;
            float down = oldnormal[0] * 0.75;

            if ((normal[0] > 0) && (oldnormal[0] > 0))
            {
                if (normal[0] > up)
                {
                    normal[0] = up;
                }
                else if (normal[0] < down)
                {
                    normal[0] = down;
                }
            }
            else if ((normal[0] < 0) && (oldnormal[0] < 0))
            {
                if (normal[0] > down)
                {
                    normal[0] = down;
                }
                else if (normal[0] < up)
                {
                    normal[0] = up;
                }
            }
            else if ((normal[0] < 0) && (oldnormal[0] > 0))
            {
                normal[0] += (oldnormal[0] - normal[0]) * 0.75;
            }
            else if ((normal[0] > 0) && (oldnormal[0] < 0))
            {
                normal[0] -= (normal[0] - oldnormal[0]) * 0.75;
            }

            up = oldnormal[1] * 1.25;
            down = oldnormal[1] * 0.75;

            if ((normal[1] > 0) && (oldnormal[1] > 0))
            {
                if (normal[1] > up)
                {
                    normal[1] = up;
                }
                else if (normal[1] < down)
                {
                    normal[1] = down;
                }
            }
            else if ((normal[1] < 0) && (oldnormal[1] < 0))
            {
                if (normal[1] > down)
                {
                    normal[1] = down;
                }
                else if (normal[1] < up)
                {
                    normal[1] = up;
                }
            }
            else if ((normal[1] <= 0) && (oldnormal[1] >= 0))
            {
                normal[1] += (oldnormal[1] - normal[1]) * 0.75;
            }
            else if ((normal[1] >= 0) && (oldnormal[1] <= 0))
            {
                normal[1] -= (normal[1] - oldnormal[1]) * 0.75;
            }
        }
        normal[2] = -1;

        normal.normalize();

        setOldnormal(normal);

        dist = pos[2] - point[0][2];
        geode = dynamic_cast<osg::Geode *>(*intersector[0]->getFirstIntersection().nodePath.end());

        if (fabs(pos[2] - point[1][2]) < fabs(dist))
        {
            dist = pos[2] - point[1][2];
            geode = dynamic_cast<osg::Geode *>(*intersector[1]->getFirstIntersection().nodePath.end());
        }
        if (fabs(pos[2] - point[2][2]) < fabs(dist))
        {
            dist = pos[2] - point[2][2];
            geode = dynamic_cast<osg::Geode *>(*intersector[2]->getFirstIntersection().nodePath.end());
        }

        if (fabs(pos[2] - point[3][2]) < fabs(dist))
        {
            dist = pos[2] - point[3][2];
            geode = dynamic_cast<osg::Geode *>(*intersector[3]->getFirstIntersection().nodePath.end());
        }
        if (fabs(pos[2] - point[4][2]) < fabs(dist))
        {
            dist = pos[2] - point[4][2];
            geode = dynamic_cast<osg::Geode *>(*intersector[4]->getFirstIntersection().nodePath.end());
        }
        if (fabs(pos[2] - point[5][2]) < fabs(dist))
        {
            dist = pos[2] - point[5][2];
            geode = dynamic_cast<osg::Geode *>(*intersector[5]->getFirstIntersection().nodePath.end());
        }

        if (geode)
        {
            std::string geodeName = geode->getName();
            if (!geodeName.empty())
            {
                if ((geodeName.find("ROAD")) != std::string::npos)
                {
                    if (SteeringWheelPlugin::plugin->sitzkiste)
                        SteeringWheelPlugin::plugin->sitzkiste->setRoadFactor(0.0); //0 == Road 1 == rough
                    if (SteeringWheelPlugin::plugin->dynamics)
                        SteeringWheelPlugin::plugin->dynamics->setRoadType(0);
                }
                else
                {
                    if (SteeringWheelPlugin::plugin->sitzkiste)
                        SteeringWheelPlugin::plugin->sitzkiste->setRoadFactor(1.0); //0 == Road 1 == rough
                    if (SteeringWheelPlugin::plugin->dynamics)
                        SteeringWheelPlugin::plugin->dynamics->setRoadType(1);
                }
            }
        }

        osg::Vec3 carNormal(carTransWorld(1, 0), carTransWorld(1, 1), carTransWorld(1, 2));
        //osg::Vec3 carNormal(0,0,1);
        if (dist > 100)
            dist = 100; // 25-10-2010
        else if (dist < -100)
            dist = -100;
        oldHeight -= dist; //Robert 11.10.10
        // oldHeight = height;
        tmp.makeTranslate(0, 0, oldHeight);
        osg::Matrix rMat;
        carNormal.normalize();
        osg::Vec3 upVec(0.0, 0.0, 1.0);
        float sprod = upVec * normal;
        if (sprod < 0)
            normal *= -1;
        sprod = upVec * normal;
        if (sprod > 0.8) // only rotate the car if the angle is not more the 45 degrees
        {
            rMat.makeRotate(carNormal, normal);
            tmp.preMult(rMat);
            moveMat = VRMLRotMat * baseMat * tmp * invBaseMat * invVRMLRotMat;
        }
        else
        {
            moveMat = VRMLRotMat * baseMat * tmp * invBaseMat * invVRMLRotMat;
        }
    }
}

// dSPACE to VRML //
//
void
PorscheRealtimeDynamics::move(VrmlNodeVehicle *vehicle)
{
    //RoadSystem* system = RoadSystem::Instance();
    //std::cout << "----------- PRD Test ---------------" << std::endl;

    // CHASSIS //
    //
    // inertial system to center of mass
    osg::Matrix inertialToCogTransform;
    inertialToCogTransform.setTrans(osg::Vec3(-DSpaceData.PosY, DSpaceData.PosZ, -DSpaceData.PosX));
    //std::cout << "DSpaceData.float_value0: " << DSpaceData.float_value0 << std::endl;
    osg::Matrix ffz1inertialToCogTransform;
    ffz1inertialToCogTransform.setTrans(osg::Vec3(-DSpaceData.ffz1_y, DSpaceData.ffz1_z, -DSpaceData.ffz1_x));

    vehicleUtil::RoadSystem::dSpace_v = DSpaceData.Geschwindigkeit * 3.6;

    //Neu 16-02-2011
    /*
   float distance;
if(coVRMSController::instance()->isMaster()) {
   distance = sqrt((DSpaceData.PosX-oX)*(DSpaceData.PosX-oX)+(DSpaceData.PosY-oY)*(DSpaceData.PosY-oY));
   cout << " >> dist: " << distance << endl;*/
    // Aktuelle Position speichern:
    //	oX = DSpaceData.PosX;
    //  oY = DSpaceData.PosY;
    //}
    //

    if (false)
    {
        /*
   //Auswertung von 25 Strahlen:
   //Koordinaten einlesen
   float height = DSpaceData.PosZ;
   osg::Vec3 p0,q0;

   // Strahlen generieren
   osg::ref_ptr<osg::LineSegment> rayA[25]= new osg::LineSegment();
   int a = 0;
   for (int i=0; i<25; i++)
   {
      p0.set(-DSpaceData.coordA[a+1],  DSpaceData.coordA[a], height+40000.0); // Strahlen +- 40 m
      q0.set(-DSpaceData.coordA[a+1],  DSpaceData.coordA[a], height-40000.0);
      rayA[i]->set(p0, q0);
      a +=2;
   }

   // Schnittpunkte auslesen
   osgUtil::IntersectVisitor visitorR;
   visitorR.setTraversalMask(Isect::Collision);
   for (int w=0; w<25; w++)
   { //Strahlen der TraversalMask hinzufuegen, die den Szenegraphen durchlaeuft
      visitorR.addLineSegment(rayA[w].get());
   }

   cover->getObjectsXform()->accept(visitorR);

   int test[25]; // Falls SP vorhanden, wird in diese Variable geschrieben
   for (int nr=0; nr<25; nr++)
   { //Ueberpruefung ob es SP gab
      test[nr]= visitorR.getNumHits(rayA[nr].get());
      cout << ">>>>>>>>SP:" << test[nr] << endl;
   }

   osgUtil::Hit hitInformationA[25];
   float hoehe[25];
   for (int h=0; h<25; h++)
   {
      if (test[h])
      {
         hitInformationA[h] = visitorR.getHitList(rayA[h].get()).front();
         hoehe[h] = hitInformationA[h].getWorldIntersectPoint()[2];
         cout << hoehe[h] << " " << h << endl;
      }
   }
   int udpcheck = toDSPACE->send(&hoehe,sizeof(hoehe));
   cout << "udpcheck: " << udpcheck << endl;
*/
    }
    ////
    /// Test mit übertragener Radposition
    if (true)
    {

        //Auswertung von 4 Strahlen:
        //Koordinaten einlesen
        float height = DSpaceData.PosZ;
        osg::Vec3 p0, q0;

        /*if(coVRMSController::instance()->isMaster())
   {
      cout << "-RadPosY " << -DSpaceData.RadPosY[0] <<" RadPosX " <<  DSpaceData.RadPosX[0] << endl;
   }*/

        // Strahlen generieren

        osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
        osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector[4];
        int a = 0;
        for (int i = 0; i < 4; i++)
        {
            p0.set(-DSpaceData.RadPosY[i], DSpaceData.RadPosX[i], height + 40000.0); // Strahlen +- 40 m
            q0.set(-DSpaceData.RadPosY[i], DSpaceData.RadPosX[i], height - 40000.0);
            intersector[i] = coIntersection::instance()->newIntersector(p0,q0);
            igroup->addIntersector(intersector[i]);
        }

        // Schnittpunkte auslesen

        osgUtil::IntersectionVisitor visitor(igroup);
        visitor.setTraversalMask(Isect::Collision);
        cover->getObjectsXform()->accept(visitor);

        

        for (int h = 0; h < 4; h++)
        {
            if (intersector[h]->containsIntersections())
            {
                rayHeight = intersector[0]->getFirstIntersection().getWorldIntersectPoint()[2];
                //cout << rayHeight << " " << h << endl;
            }
        }
        /*
   int udpcheck = toDSPACE->send(&hoehe[0],sizeof(hoehe));
   cout << "udpcheck: " << udpcheck << endl;
*/
    }
    ////

    osg::Quat inertialToCogRotation(
        DSpaceData.PitchAngle, osg::Vec3(-1, 0, 0),
        DSpaceData.RollAngle, osg::Vec3(0, 0, -1),
        DSpaceData.YawAngle, osg::Vec3(0, 1, 0));
    inertialToCogTransform.setRotate(inertialToCogRotation);

    osg::Quat ffz1inertialToCogRotation(
        DSpaceData.ffz1_pitch, osg::Vec3(-1, 0, 0),
        DSpaceData.ffz1_roll, osg::Vec3(0, 0, -1),
        DSpaceData.ffz1_yaw, osg::Vec3(0, 1, 0));
    ffz1inertialToCogTransform.setRotate(ffz1inertialToCogRotation);

    // total
    chassisTransform = inertialToCogTransform;

    ffz1chassisTransform = ffz1inertialToCogTransform;

    dSpacePose = chassisTransform;

    //cerr << "height:" << DSpaceData.PosZ << endl;

    // WHEELS //

    //
    // inertial system to wheels
    osg::Matrix inertialToWheelTransform[4];
    for (int i = 0; i < 4; i++)
    {
        inertialToWheelTransform[i].setTrans(osg::Vec3(-DSpaceData.RadPosY[i], DSpaceData.RadPosZ[i], -DSpaceData.RadPosX[i]));
        osg::Quat inertialToWheelRotation(
            DSpaceData.WheelRotation[i] + DSpaceData.PitchAngle, osg::Vec3(-1, 0, 0),
            DSpaceData.WheelCamber[i] + DSpaceData.RollAngle, osg::Vec3(0, 0, -1),
            DSpaceData.WheelAngle[i] + DSpaceData.YawAngle, osg::Vec3(0, 1, 0));
        inertialToWheelTransform[i].setRotate(inertialToWheelRotation);
    }

    // inertial system to ffz-wheels
    osg::Matrix ffz1inertialToWheelTransform[4];
    for (int i = 0; i < 4; i++)
    {
        ffz1inertialToWheelTransform[i].setTrans(0, DSpaceData.ffz1_RadPosZ[i], 0);
        if (i < 2)
        {
            osg::Quat ffz1inertialToWheelRotation(
                DSpaceData.ffz1_WheelRotation + DSpaceData.ffz1_pitch, osg::Vec3(0, 0, -1), //Rotationsachse vertauscht!
                DSpaceData.ffz1_roll, osg::Vec3(-1, 0, 0),
                DSpaceData.ffz1_WheelAngle, osg::Vec3(0, 1, 0));
            ffz1inertialToWheelTransform[i].setRotate(ffz1inertialToWheelRotation);
        }

        else
        {
            osg::Quat ffz1inertialToWheelRotation(
                DSpaceData.ffz1_WheelRotation + DSpaceData.ffz1_pitch, osg::Vec3(0, 0, -1), //Rotationsachse vertauscht!
                DSpaceData.ffz1_roll, osg::Vec3(-1, 0, 0),
                0, osg::Vec3(0, 1, 0));
            ffz1inertialToWheelTransform[i].setRotate(ffz1inertialToWheelRotation);
        }
    }

// OFFSET //
//
// an offset to the dSPACE translation values, to place the car in the scene
/*
only absolute vaulues from the realtime dynamics, reset will not work.
#if 1
    chassisTransform.setRotate(chassisTransform.getRotate() * dSpaceOrigin.getRotate());
    chassisTransform.setTrans(dSpaceOrigin.getTrans() + osg::Matrix(dSpaceOrigin.getRotate().inverse()) * chassisTransform.getTrans());

    ffz1chassisTransform.setRotate(ffz1chassisTransform.getRotate() * dSpaceOrigin.getRotate());
    ffz1chassisTransform.setTrans(dSpaceOrigin.getTrans() + osg::Matrix(dSpaceOrigin.getRotate().inverse()) * ffz1chassisTransform.getTrans());

    for (int i = 0; i < 4; i++)
    {
        inertialToWheelTransform[i].setRotate(inertialToWheelTransform[i].getRotate() * dSpaceOrigin.getRotate());
        inertialToWheelTransform[i].setTrans(dSpaceOrigin.getTrans() + osg::Matrix(dSpaceOrigin.getRotate().inverse()) * inertialToWheelTransform[i].getTrans());

        //Räder von Ghostfahrzeug
        ffz1inertialToWheelTransform[i].setRotate(ffz1inertialToWheelTransform[i].getRotate() * dSpaceOrigin.getRotate());
        ffz1inertialToWheelTransform[i].setTrans(dSpaceOrigin.getTrans() + osg::Matrix(dSpaceOrigin.getRotate().inverse()) * ffz1inertialToWheelTransform[i].getTrans());
    }
#else
    osg::Vec3 deltaC(-dSpaceReference.getTrans() + dSpacePose.getTrans());
    osg::Quat deltaQ(dSpaceReference.getRotate().inverse() * dSpacePose.getRotate());
    chassisTransform.setTrans(targetReference.getTrans() + osg::Matrix(targetReference.getRotate().inverse()) * osg::Matrix(dSpaceReference.getRotate()) * deltaC);
    chassisTransform.setRotate(targetReference.getRotate() * deltaQ);
#endif
*/

    // SNAP TO GROUND //
    //
    /*if (vehicle->followTerrain())
    {
        // calculate matrix
        osg::Matrix moveMat;
        moveToStreet(chassisTransform, moveMat);

        // apply matrix
        chassisTransform.postMult(moveMat);
        for (int i = 0; i < 4; i++)
        {
            inertialToWheelTransform[i].postMult(moveMat);
        }
    }
    else
    {
        // DIRTY HACK FOR PRE-CRASH STUDY //
        //chassisTransform.setTrans(chassisTransform.getTrans()[0], chassisTransform.getTrans()[1], chassisTransform.getTrans()[2]);
        //ffz1chassisTransform.setTrans(ffz1chassisTransform.getTrans()[0], ffz1chassisTransform.getTrans()[1], ffz1chassisTransform.getTrans()[2]);
    }
    */

    // UPDATE VRML //
    //
    //cerr << "m: " << chassisTransform.getTrans()[0] << " " << chassisTransform.getTrans()[1] << " " << chassisTransform.getTrans()[2] << " " << endl;
    //cerr << "m: " << chassisTransform(0,0) << " " << chassisTransform.getTrans()[1] << " " << chassisTransform.getTrans()[2] << " " << endl;
   /* fprintf(stderr,"chassisTransform\n");
    fprintf(stderr," %3.3f %3.3f %3.3f %3.3f\n", chassisTransform(0,0), chassisTransform(1,0), chassisTransform(2,0), chassisTransform(3,0));
    fprintf(stderr," %3.3f %3.3f %3.3f %3.3f\n", chassisTransform(0,1), chassisTransform(1,1), chassisTransform(2,1), chassisTransform(3,1));
    fprintf(stderr," %3.3f %3.3f %3.3f %3.3f\n", chassisTransform(0,2), chassisTransform(1,2), chassisTransform(2,2), chassisTransform(3,2));
    fprintf(stderr," %3.3f %3.3f %3.3f %3.3f\n", chassisTransform(0,3), chassisTransform(1,3), chassisTransform(2,3), chassisTransform(3,3));*/
    vehicle->setVRMLVehicle(chassisTransform);
    vehicle->setVRMLVehicleFFZBody(ffz1chassisTransform);
    vehicle->setVRMLVehicleBody(chassisTransform);
    vehicle->setVRMLVehicleFrontWheels(inertialToWheelTransform[0], inertialToWheelTransform[1]);
    vehicle->setVRMLVehicleRearWheels(inertialToWheelTransform[2], inertialToWheelTransform[3]);
    vehicle->setVRMLAdditionalData(DSpaceData.float_value0, DSpaceData.float_value1, DSpaceData.float_value2, DSpaceData.float_value3, DSpaceData.float_value4, DSpaceData.float_value5, DSpaceData.float_value6, DSpaceData.float_value7, DSpaceData.float_value8, DSpaceData.float_value9, DSpaceData.int_value0, DSpaceData.int_value1, DSpaceData.int_value2, DSpaceData.int_value3, DSpaceData.int_value4, DSpaceData.int_value5, DSpaceData.int_value6, DSpaceData.int_value7, DSpaceData.int_value8, DSpaceData.int_value9);

    vehicle->setVRMLVehicleFFZFrontWheels(ffz1inertialToWheelTransform[0], ffz1inertialToWheelTransform[1]);
    vehicle->setVRMLVehicleFFZRearWheels(ffz1inertialToWheelTransform[2], ffz1inertialToWheelTransform[3]);

    // ??? //
    //
    //double heightFL, heightFR, heightRL, heightRR;
    //vehicle->getGroundDistance(inertialToWheelFLTransform, inertialToWheelFRTransform, inertialToWheelRLTransform, inertialToWheelRRTransform, heightFL, heightFR, heightRL, heightRR);
    //std::cerr << "Heights: " << heightFL << ", " << heightFR << ", " << heightRL << ", " << heightRR << std::endl;
}

void
PorscheRealtimeDynamics::setVehicleTransformationOffset(const osg::Matrix &targetPose)
{

#if 1
    // dSPACE //
    //
    osg::Vec3 dSpaceLoc = dSpacePose.getTrans();
    osg::Quat dSpaceRot = dSpacePose.getRotate();

    // TARGET //
    //
    osg::Vec3 targetLoc(targetPose.getTrans());
    osg::Quat targetRot(targetPose.getRotate());

    // ORIGIN //
    //
    dSpaceOrigin.setRotate(targetRot * dSpaceRot.inverse());
    dSpaceOrigin.setTrans(targetLoc - osg::Matrix(dSpaceOrigin.getRotate().inverse()) * dSpaceLoc);

#else
    // dSPACE //
    //
    dSpaceReference.setTrans(dSpacePose.getTrans());
    dSpaceReference.setRotate(dSpacePose.getRotate());

    targetReference.setTrans(targetPose.getTrans());
    targetReference.setRotate(targetPose.getRotate());

#endif

    resetState();
}

void
PorscheRealtimeDynamics::setVehicleTransformation(const osg::Matrix &m)
{
    oldHeight = m.getTrans()[2];
}

void
PorscheRealtimeDynamics::resetState()
{
    outputData[0] = 1;
    oldHeight = 0;
    chassisTransform.makeIdentity();
    ffz1chassisTransform.makeIdentity();
}

#if 1
/** VRML node for the Porsche Virtueller Fahrerplatz.
 *
 *
 *
 *
 *
*/

void VrmlNodePorscheVFP::initFields(VrmlNodePorscheVFP *node, VrmlNodeType *t)
{
    VrmlNodeChild::initFields(node, t);
    initFieldsHelper(node, t, 
        exposedField("targetIP", node->d_targetIP),
        exposedField("targetPort", node->d_targetPort),
        exposedField("messageID", node->d_messageID),
        exposedField("messageIDIn", node->d_messageIDIn));
    for (int i = 0; i < NUM_FIELDS; i++)
    {
        initFieldsHelper(node, t, 
            exposedField("floatValue" + std::to_string(i), node->d_floatValues[i]),
            exposedField("floatValueIn" + std::to_string(i), node->d_floatValueIn[i]),
            exposedField("intValue" + std::to_string(i), node->d_intValues[i]),
            exposedField("intValueIn" + std::to_string(i), node->d_intValueIn[i]));
    }

    if(t)
    {
        t->addEventIn("connectToServer", VrmlField::SFTIME);
        t->addEventIn("sendData", VrmlField::SFTIME);

        t->addEventOut("receivedData", VrmlField::SFTIME);
        t->addEventOut("connectionEstablished", VrmlField::SFTIME);
        t->addEventOut("noConnection", VrmlField::SFTIME); // TODO
        t->addEventOut("sendError", VrmlField::SFTIME);        
    }


}

const char *VrmlNodePorscheVFP::name()
{
    return "PorscheVFP";
}

/** Constructor.
*/
VrmlNodePorscheVFP::VrmlNodePorscheVFP(VrmlScene *scene)
    : VrmlNodeChild(scene, name())
    , clientConn_(NULL)
    , d_targetIP("")
    , d_targetPort(0)
    ,
    // 		d_localPort(0),
    d_messageID(0)
    , d_messageIDIn(0)
    , d_lastReceiveTime(0.0)
{
    std::fill(d_floatValues.begin(), d_floatValues.end(), 0.0f);
    std::fill(d_floatValueIn.begin(), d_floatValueIn.end(), 0.0f);
    std::fill(d_intValues.begin(), d_intValues.end(), 0);
    std::fill(d_intValueIn.begin(), d_intValueIn.end(), 0);
}

/** Copy Constructor.
*/
VrmlNodePorscheVFP::VrmlNodePorscheVFP(const VrmlNodePorscheVFP &n)
    : VrmlNodeChild(n)
    , clientConn_(NULL)
    , d_targetIP(n.d_targetIP)
    , d_targetPort(n.d_targetPort)
    ,
    // 		d_localPort(n.d_localPort),
    d_messageID(n.d_messageID)
    , d_floatValues(n.d_floatValues)
    , d_intValues(n.d_intValues)
    , d_messageIDIn(n.d_messageIDIn)
    , d_floatValueIn(n.d_floatValueIn)
    , d_intValueIn(n.d_intValueIn)

    , d_lastReceiveTime(n.d_lastReceiveTime)

{
}

/** Render.
*/
void
VrmlNodePorscheVFP::render(Viewer *)
{

    bool dataReceived = false;
    if (coVRMSController::instance()->isMaster())
    {
        dataReceived = getTCPData();
    }

    if (coVRMSController::instance()->isMaster())
    {
        coVRMSController::instance()->sendSlaves((char *)&dataReceived, sizeof(dataReceived));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&dataReceived, sizeof(dataReceived));
    }

    if (dataReceived)
    {
        syncTCPData(); // TODO: send only if there is data
    }

    clearModified();
}

/** Handle eventIn.
*/
void
VrmlNodePorscheVFP::eventIn(double timeStamp, const char *eventName, const VrmlField *fieldValue)
{
    if (strcmp(eventName, "sendData") == 0)
    {
        if (coVRMSController::instance()->isMaster())
            sendTCPData();
    }

    else if (strcmp(eventName, "connectToServer") == 0)
    {
        connectToServer(d_targetIP.get(), d_targetPort.get());
    }

    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

void
VrmlNodePorscheVFP::connectToServer(std::string targetIP, int targetPort /*, int localPort*/)
{
    // Only for Master //
    //
    if (!(coVRMSController::instance()->isMaster()))
        return;

    // Already opened? //
    //
    if (clientConn_ && clientConn_->is_connected())
    {
        std::cout << "\nWARNING: CONNECTION ALREADY ESTABLISHED, Please stop reconnecting!\n" << std::endl;
        return;
    }

    // Check for reasonable values //
    //
    if (targetIP == "" || targetPort == 0 /*|| localPort == 0*/)
    {
        std::cout << "\nWARNING: TRYING TO CONNECT TO SERVER BUT NO SERVER SPECIFIED!\n" << std::endl;
        return;
    }

    // SETUP CONNECTION //
    //
    serverHost_ = new Host(targetIP.c_str(), true); // numeric = true: ip, not domain
    if ((serverHost_ == NULL))
    {
        std::cout << "WARNING: COULD NOT PARSE SERVER IP: " << targetIP << std::endl;
        delete serverHost_;
        serverHost_ = NULL;
        return;
    }

    clientConn_ = new SimpleClientConnection(serverHost_, targetPort, 0);
    if (!clientConn_->is_connected())
    {
        // could not open server //
        std::cout << "\nWARNING: Failed to connect to Porsche PC: " << targetIP << " on port " << targetPort << "\n" << std::endl;
        delete serverHost_;
        serverHost_ = NULL;
        delete clientConn_;
        clientConn_ = NULL;
    }
    else
    {
        // could open server //
        std::cout << "\nConnected to Porsche PC: " << targetIP << " on port " << targetPort << "\n" << std::endl;

        double timeStamp = System::the->time();
        VrmlSFTime d_time(timeStamp);
        eventOut(timeStamp, "connectionEstablished", d_time);
    }
    return;
}

void
VrmlNodePorscheVFP::sendTCPData()
{
    if (clientConn_ && clientConn_->is_connected())
    {
        //std::cout << "sending: " << d_messageID << ", " << d_floatValue0 << " " << d_floatValue1 << " " << d_floatValue2  << std::endl;

        // Create a data package with the vrml data and send it //
        //
        VFPdataPackage outData;

        outData.messageID = d_messageID.get();
        for (size_t i = 0; i < NUM_FIELDS; i++)
        {
            outData.floatValues[i] = d_floatValues[i].get();
            outData.intValues[i] = d_intValues[i].get();
        }
        clientConn_->send(&outData, sizeof(outData));
    }
    else
    {
        double timeStamp = System::the->time();
        VrmlSFTime d_time(timeStamp);
        eventOut(timeStamp, "sendError", d_time);

        std::cout << "Error 1003031331: Can't send data. Not connected to KMS!" << std::endl;
    }
}

bool
VrmlNodePorscheVFP::readTCPData(void *buf, unsigned int numBytes)
{
    //std::cout << "read bytes: " << numBytes << std::endl;
    unsigned int stillToRead = numBytes;
    unsigned int alreadyRead = 0;
    int readBytes = 0;
    while (alreadyRead < numBytes)
    {
        readBytes = clientConn_->getSocket()->Read(((unsigned char *)buf) + alreadyRead, stillToRead);
        if (readBytes < 0)
        {
            std::cout << "Error 1003031335: KMS: error reading data from socket" << std::endl;
            return false;
        }
        alreadyRead += readBytes;
        stillToRead = numBytes - alreadyRead;
    }
    //std::cout << "message: " << (int)(buf[4]) << (int)(buf[5]) << (int)(buf[6]) << (int)(buf[7]) << std::endl;
    //std::cout << "message: " << inData.floatValues[0] << std::endl;
    return true;
}

bool
VrmlNodePorscheVFP::getTCPData()
{
    if (!coVRMSController::instance()->isMaster())
        return false;

    if (!(clientConn_ && clientConn_->is_connected()))
        return false;

    if (clientConn_->check_for_input())
    {
        // Data to Master //
        //
        int messageID[1];
        if (!readTCPData(messageID, sizeof(messageID)))
        {
            std::cout << "Error 1003031334: KMS: error reading the data from socket" << std::endl;
            return false;
        }

        int numFloats = 16;
        float *floatVals = new float[numFloats];
        if (!readTCPData(floatVals, numFloats * sizeof(float)))
        {
            std::cout << "Error 1003031333: KMS: error reading float data from socket" << std::endl;
            return false;
        }

        int numInts = 16;
        int *intVals = new int[numInts];
        if (!readTCPData(intVals, numInts * sizeof(int)))
        {
            std::cout << "Error 1003031332: KMS: error reading int data from socket" << std::endl;
            return false;
        }

        // Save Data from KMS in fromKMS (Master) //
        //
        fromKMS.messageID = messageID[0];
        for (int i = 0; i < 16; i++)
        {
            fromKMS.floatValues[i] = floatVals[i];
            fromKMS.intValues[i] = intVals[i];
        }
        delete[] floatVals;
        delete[] intVals;
        return true;
    }
    if (clientConn_->check_for_input())
    {
        std::cout << "SECOND INPUT DETECTED!" << std::endl;
    }

    return false;
}

void
VrmlNodePorscheVFP::syncTCPData()
{
    // Synchronize Master and Slaves /
    //
    if (coVRMSController::instance()->isMaster())
    {
        memcpy((void *)&fromMaster, (void *)&fromKMS, sizeof(fromMaster));
        coVRMSController::instance()->sendSlaves((char *)&fromMaster, sizeof(fromMaster));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&fromMaster, sizeof(fromMaster));
    }

    // Synchronize VRML //
    //
    d_messageIDIn = fromMaster.messageID;

    for (size_t i = 0; i < NUM_FIELDS; i++)
    {
        d_floatValueIn[i] = fromMaster.floatValues[i];
        d_intValueIn[i] = fromMaster.intValues[i];
    }

    // Events //
    //
    double timeStamp = System::the->time();

    eventOut(timeStamp, "messageIDIn", d_messageIDIn);

    for (size_t i = 0; i < NUM_FIELDS; i++)
    {
        eventOut(timeStamp, "floatValueIn0", d_floatValueIn[i]);
        eventOut(timeStamp, "intValueIn0", d_intValueIn[i]);

    }

    d_lastReceiveTime.set(timeStamp);
    eventOut(timeStamp, "receivedData", d_lastReceiveTime);

    // 	std::cout << d_messageIDIn << " " << d_floatValueIn0 << " " << d_floatValueIn1 << " " << d_floatValueIn2 << std::endl;
}

#endif
void
PorscheRealtimeDynamics::initOldnormal()
{
    osg::Vec3 init(0, 0, 0);
    oldnormal = init;
}

osg::Vec3
PorscheRealtimeDynamics::getOldnormal()
{
    return oldnormal;
}

void
PorscheRealtimeDynamics::setOldnormal(osg::Vec3 newNormal)
{
    oldnormal = newNormal;
}

//
