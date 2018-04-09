/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PorscheRealtimeDynamics.h"


#include <OpenVRUI/osg/mathUtils.h>
#include <config/CoviseConfig.h>
#include "SteeringWheel.h"

#include <cover/coVRTui.h>

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

    // Front-Right segment
    p0.set(pos[0] + 2000, pos[1] + 1500, pos[2] + 1500.0); // 1.5 m above actual position and ...
    q0.set(pos[0] + 2000, pos[1] + 1500, pos[2] - 40000.0); // 40 m under actual position
    osg::ref_ptr<osg::LineSegment> ray1 = new osg::LineSegment();
    ray1->set(p0, q0);

    // Back-Center segment
    p0.set(pos[0], pos[1] - 1500, pos[2] + 1500.0);
    q0.set(pos[0], pos[1] - 1500, pos[2] - 40000.0);
    osg::ref_ptr<osg::LineSegment> ray2 = new osg::LineSegment();
    ray2->set(p0, q0);

    // Front-Left segment
    p0.set(pos[0] - 2000, pos[1] + 1500, pos[2] + 1500.0);
    q0.set(pos[0] - 2000, pos[1] + 1500, pos[2] - 40000.0);
    osg::ref_ptr<osg::LineSegment> ray3 = new osg::LineSegment();
    ray3->set(p0, q0);

    //----------------------//
    //  ... for plane 2     //
    //----------------------//

    // Front-center segment
    p0.set(pos[0], pos[1] + 1500, pos[2] + 1500.0);
    q0.set(pos[0], pos[1] + 1500, pos[2] - 40000.0);
    osg::ref_ptr<osg::LineSegment> ray4 = new osg::LineSegment();
    ray4->set(p0, q0);

    // Back-Right segment
    p0.set(pos[0] + 2000, pos[1] - 1500, pos[2] + 1500.0);
    q0.set(pos[0] + 2000, pos[1] - 1500, pos[2] - 40000.0);
    osg::ref_ptr<osg::LineSegment> ray5 = new osg::LineSegment();
    ray5->set(p0, q0);

    // Back-Left segment
    p0.set(pos[0] - 2000, pos[1] - 1500, pos[2] + 1500.0);
    q0.set(pos[0] - 2000, pos[1] - 1500, pos[2] - 40000.0);
    osg::ref_ptr<osg::LineSegment> ray6 = new osg::LineSegment();
    ray6->set(p0, q0);

    osgUtil::IntersectVisitor visitor;
    visitor.setTraversalMask(Isect::Collision);
    visitor.addLineSegment(ray1.get());
    visitor.addLineSegment(ray2.get());
    visitor.addLineSegment(ray3.get());
    visitor.addLineSegment(ray4.get());
    visitor.addLineSegment(ray5.get());
    visitor.addLineSegment(ray6.get());

    cover->getObjectsXform()->accept(visitor);
    int num1 = visitor.getNumHits(ray1.get());
    int num2 = visitor.getNumHits(ray2.get());
    int num3 = visitor.getNumHits(ray3.get());

    int num4 = visitor.getNumHits(ray4.get());
    int num5 = visitor.getNumHits(ray5.get());
    int num6 = visitor.getNumHits(ray6.get());

    ////////////////////////////////////////
    // 		   Intersection check		  //
    ////////////////////////////////////////

    if (num1 || num2 || num3 || num4 || num5 || num6)
    {
        osgUtil::Hit hitInformation1;
        osgUtil::Hit hitInformation2;
        osgUtil::Hit hitInformation3;
        osgUtil::Hit hitInformation4;
        osgUtil::Hit hitInformation5;
        osgUtil::Hit hitInformation6;

        if (num1)
            hitInformation1 = visitor.getHitList(ray1.get()).front();
        if (num2)
            hitInformation2 = visitor.getHitList(ray2.get()).front();
        if (num3)
            hitInformation3 = visitor.getHitList(ray3.get()).front();
        if (num4)
            hitInformation4 = visitor.getHitList(ray4.get()).front();
        if (num5)
            hitInformation5 = visitor.getHitList(ray5.get()).front();
        if (num6)
            hitInformation6 = visitor.getHitList(ray6.get()).front();

        float dist = 0.0;
        osg::Vec3 normal(0, 0, 1);
        osg::Vec3 oldnormal(0, 0, 1);
        osg::Vec3 realnormal(0, 0, 1);
        osg::Geode *geode = NULL;

        /////////////////////////////////////////////////////////////////////////
        //          Creating a normal out of the normals of two planes         //
        /////////////////////////////////////////////////////////////////////////

        osg::Vec3 point1(0, 0, 0);
        osg::Vec3 point2(0, 0, 0);
        osg::Vec3 point3(0, 0, 0);

        osg::Vec3 point4(0, 0, 0);
        osg::Vec3 point5(0, 0, 0);
        osg::Vec3 point6(0, 0, 0);

        // Interseciton points plane1
        point1 = hitInformation1.getWorldIntersectPoint();
        point2 = hitInformation2.getWorldIntersectPoint();
        point3 = hitInformation3.getWorldIntersectPoint();
        // Intersection points plane2
        point4 = hitInformation4.getWorldIntersectPoint();
        point5 = hitInformation5.getWorldIntersectPoint();
        point6 = hitInformation6.getWorldIntersectPoint();

        //------------------------------//
        //         create planes        //
        //------------------------------//

        osg::Plane plane(point1, point2, point3);
        osg::Plane plane2(point4, point5, point6);

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

        dist = pos[2] - point1[2];
        geode = hitInformation1.getGeode();

        if (fabs(pos[2] - point2[2]) < fabs(dist))
        {
            dist = pos[2] - point2[2];
            geode = hitInformation2.getGeode();
        }
        if (fabs(pos[2] - point3[2]) < fabs(dist))
        {
            dist = pos[2] - point3[2];
            geode = hitInformation3.getGeode();
        }

        if (fabs(pos[2] - point4[2]) < fabs(dist))
        {
            dist = pos[2] - point4[2];
            geode = hitInformation4.getGeode();
        }
        if (fabs(pos[2] - point5[2]) < fabs(dist))
        {
            dist = pos[2] - point5[2];
            geode = hitInformation5.getGeode();
        }
        if (fabs(pos[2] - point6[2]) < fabs(dist))
        {
            dist = pos[2] - point6[2];
            geode = hitInformation6.getGeode();
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

    RoadSystem::dSpace_v = DSpaceData.Geschwindigkeit * 3.6;

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

        osg::ref_ptr<osg::LineSegment> rayA[4];
        int a = 0;
        for (int i = 0; i < 4; i++)
        {
            rayA[i] = new osg::LineSegment();
            p0.set(-DSpaceData.RadPosY[i], DSpaceData.RadPosX[i], height + 40000.0); // Strahlen +- 40 m
            q0.set(-DSpaceData.RadPosY[i], DSpaceData.RadPosX[i], height - 40000.0);
            rayA[i]->set(p0, q0);
        }

        // Schnittpunkte auslesen
        osgUtil::IntersectVisitor visitorR;
        visitorR.setTraversalMask(Isect::Collision);
        for (int w = 0; w < 4; w++)
        { //Strahlen der TraversalMask hinzufuegen, die den Szenegraphen durchlaeuft
            visitorR.addLineSegment(rayA[w].get());
        }

        cover->getObjectsXform()->accept(visitorR);

        int test[4]; // Falls SP vorhanden, wird in diese Variable geschrieben
        for (int nr = 0; nr < 4; nr++)
        { //Ueberpruefung ob es SP gab
            test[nr] = visitorR.getNumHits(rayA[nr].get());
            //cout << ">>>>>>>>SP:" << test[nr] << "nr " << nr << endl;
        }

        osgUtil::Hit hitInformationA[4];
        for (int h = 0; h < 4; h++)
        {
            if (test[h])
            {
                hitInformationA[h] = visitorR.getHitList(rayA[h].get()).front();
                rayHeight = hitInformationA[h].getWorldIntersectPoint()[2];
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

/** Return a new VrmlNode.
*/
static VrmlNode *creatorPorscheVFP(VrmlScene *s) { return new VrmlNodePorscheVFP(s); }

/** Define fields and type.
*/
VrmlNodeType *
VrmlNodePorscheVFP::defineType(VrmlNodeType *t)
{

    // Type //
    //
    static VrmlNodeType *st = 0;
    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("PorscheVFP", creatorPorscheVFP);
    }
    VrmlNodeChild::defineType(t); // Parent class

    // Fields //
    //
    t->addEventIn("connectToServer", VrmlField::SFTIME);
    t->addEventIn("sendData", VrmlField::SFTIME);

    t->addEventOut("receivedData", VrmlField::SFTIME);
    t->addEventOut("connectionEstablished", VrmlField::SFTIME);
    t->addEventOut("noConnection", VrmlField::SFTIME); // TODO
    t->addEventOut("sendError", VrmlField::SFTIME);

    t->addExposedField("targetIP", VrmlField::SFSTRING);
    t->addExposedField("targetPort", VrmlField::SFINT32);
    // 	t->addExposedField("localPort", VrmlField::SFINT32);

    // Out Data //
    //
    t->addExposedField("messageID", VrmlField::SFINT32);

    t->addExposedField("floatValue0", VrmlField::SFFLOAT);
    t->addExposedField("floatValue1", VrmlField::SFFLOAT);
    t->addExposedField("floatValue2", VrmlField::SFFLOAT);
    t->addExposedField("floatValue3", VrmlField::SFFLOAT);
    t->addExposedField("floatValue4", VrmlField::SFFLOAT);
    t->addExposedField("floatValue5", VrmlField::SFFLOAT);
    t->addExposedField("floatValue6", VrmlField::SFFLOAT);
    t->addExposedField("floatValue7", VrmlField::SFFLOAT);
    t->addExposedField("floatValue8", VrmlField::SFFLOAT);
    t->addExposedField("floatValue9", VrmlField::SFFLOAT);
    t->addExposedField("floatValue10", VrmlField::SFFLOAT);
    t->addExposedField("floatValue11", VrmlField::SFFLOAT);
    t->addExposedField("floatValue12", VrmlField::SFFLOAT);
    t->addExposedField("floatValue13", VrmlField::SFFLOAT);
    t->addExposedField("floatValue14", VrmlField::SFFLOAT);
    t->addExposedField("floatValue15", VrmlField::SFFLOAT);

    t->addExposedField("intValue0", VrmlField::SFINT32);
    t->addExposedField("intValue1", VrmlField::SFINT32);
    t->addExposedField("intValue2", VrmlField::SFINT32);
    t->addExposedField("intValue3", VrmlField::SFINT32);
    t->addExposedField("intValue4", VrmlField::SFINT32);
    t->addExposedField("intValue5", VrmlField::SFINT32);
    t->addExposedField("intValue6", VrmlField::SFINT32);
    t->addExposedField("intValue7", VrmlField::SFINT32);
    t->addExposedField("intValue8", VrmlField::SFINT32);
    t->addExposedField("intValue9", VrmlField::SFINT32);
    t->addExposedField("intValue10", VrmlField::SFINT32);
    t->addExposedField("intValue11", VrmlField::SFINT32);
    t->addExposedField("intValue12", VrmlField::SFINT32);
    t->addExposedField("intValue13", VrmlField::SFINT32);
    t->addExposedField("intValue14", VrmlField::SFINT32);
    t->addExposedField("intValue15", VrmlField::SFINT32);

    // In Data //
    //
    t->addExposedField("messageIDIn", VrmlField::SFINT32);

    t->addExposedField("floatValueIn0", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn1", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn2", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn3", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn4", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn5", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn6", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn7", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn8", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn9", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn10", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn11", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn12", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn13", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn14", VrmlField::SFFLOAT);
    t->addExposedField("floatValueIn15", VrmlField::SFFLOAT);

    t->addExposedField("intValueIn0", VrmlField::SFINT32);
    t->addExposedField("intValueIn1", VrmlField::SFINT32);
    t->addExposedField("intValueIn2", VrmlField::SFINT32);
    t->addExposedField("intValueIn3", VrmlField::SFINT32);
    t->addExposedField("intValueIn4", VrmlField::SFINT32);
    t->addExposedField("intValueIn5", VrmlField::SFINT32);
    t->addExposedField("intValueIn6", VrmlField::SFINT32);
    t->addExposedField("intValueIn7", VrmlField::SFINT32);
    t->addExposedField("intValueIn8", VrmlField::SFINT32);
    t->addExposedField("intValueIn9", VrmlField::SFINT32);
    t->addExposedField("intValueIn10", VrmlField::SFINT32);
    t->addExposedField("intValueIn11", VrmlField::SFINT32);
    t->addExposedField("intValueIn12", VrmlField::SFINT32);
    t->addExposedField("intValueIn13", VrmlField::SFINT32);
    t->addExposedField("intValueIn14", VrmlField::SFINT32);
    t->addExposedField("intValueIn15", VrmlField::SFINT32);

    return t;
}

/** Calls defineType().
*/
VrmlNodeType *
VrmlNodePorscheVFP::nodeType() const
{
    return defineType(0);
}

/** Constructor.
*/
VrmlNodePorscheVFP::VrmlNodePorscheVFP(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , clientConn_(NULL)
    , d_targetIP("")
    , d_targetPort(0)
    ,
    // 		d_localPort(0),
    d_messageID(0)
    , d_floatValue0(0.0f)
    , d_floatValue1(0.0f)
    , d_floatValue2(0.0f)
    , d_floatValue3(0.0f)
    , d_floatValue4(0.0f)
    , d_floatValue5(0.0f)
    , d_floatValue6(0.0f)
    , d_floatValue7(0.0f)
    , d_floatValue8(0.0f)
    , d_floatValue9(0.0f)
    , d_floatValue10(0.0f)
    , d_floatValue11(0.0f)
    , d_floatValue12(0.0f)
    , d_floatValue13(0.0f)
    , d_floatValue14(0.0f)
    , d_floatValue15(0.0f)
    , d_intValue0(0)
    , d_intValue1(0)
    , d_intValue2(0)
    , d_intValue3(0)
    , d_intValue4(0)
    , d_intValue5(0)
    , d_intValue6(0)
    , d_intValue7(0)
    , d_intValue8(0)
    , d_intValue9(0)
    , d_intValue10(0)
    , d_intValue11(0)
    , d_intValue12(0)
    , d_intValue13(0)
    , d_intValue14(0)
    , d_intValue15(0)
    , d_messageIDIn(0)
    , d_floatValueIn0(0.0f)
    , d_floatValueIn1(0.0f)
    , d_floatValueIn2(0.0f)
    , d_floatValueIn3(0.0f)
    , d_floatValueIn4(0.0f)
    , d_floatValueIn5(0.0f)
    , d_floatValueIn6(0.0f)
    , d_floatValueIn7(0.0f)
    , d_floatValueIn8(0.0f)
    , d_floatValueIn9(0.0f)
    , d_floatValueIn10(0.0f)
    , d_floatValueIn11(0.0f)
    , d_floatValueIn12(0.0f)
    , d_floatValueIn13(0.0f)
    , d_floatValueIn14(0.0f)
    , d_floatValueIn15(0.0f)
    , d_intValueIn0(0)
    , d_intValueIn1(0)
    , d_intValueIn2(0)
    , d_intValueIn3(0)
    , d_intValueIn4(0)
    , d_intValueIn5(0)
    , d_intValueIn6(0)
    , d_intValueIn7(0)
    , d_intValueIn8(0)
    , d_intValueIn9(0)
    , d_intValueIn10(0)
    , d_intValueIn11(0)
    , d_intValueIn12(0)
    , d_intValueIn13(0)
    , d_intValueIn14(0)
    , d_intValueIn15(0)
    , d_lastReceiveTime(0.0)
{
}

/** Copy Constructor.
*/
VrmlNodePorscheVFP::VrmlNodePorscheVFP(const VrmlNodePorscheVFP &n)
    : VrmlNodeChild(n.d_scene)
    , clientConn_(NULL)
    , d_targetIP(n.d_targetIP)
    , d_targetPort(n.d_targetPort)
    ,
    // 		d_localPort(n.d_localPort),
    d_messageID(n.d_messageID)
    , d_floatValue0(n.d_floatValue0)
    , d_floatValue1(n.d_floatValue1)
    , d_floatValue2(n.d_floatValue2)
    , d_floatValue3(n.d_floatValue3)
    , d_floatValue4(n.d_floatValue4)
    , d_floatValue5(n.d_floatValue5)
    , d_floatValue6(n.d_floatValue6)
    , d_floatValue7(n.d_floatValue7)
    , d_floatValue8(n.d_floatValue8)
    , d_floatValue9(n.d_floatValue9)
    , d_floatValue10(n.d_floatValue10)
    , d_floatValue11(n.d_floatValue11)
    , d_floatValue12(n.d_floatValue12)
    , d_floatValue13(n.d_floatValue13)
    , d_floatValue14(n.d_floatValue14)
    , d_floatValue15(n.d_floatValue15)
    , d_intValue0(n.d_intValue0)
    , d_intValue1(n.d_intValue1)
    , d_intValue2(n.d_intValue2)
    , d_intValue3(n.d_intValue3)
    , d_intValue4(n.d_intValue4)
    , d_intValue5(n.d_intValue5)
    , d_intValue6(n.d_intValue6)
    , d_intValue7(n.d_intValue7)
    , d_intValue8(n.d_intValue8)
    , d_intValue9(n.d_intValue9)
    , d_intValue10(n.d_intValue10)
    , d_intValue11(n.d_intValue11)
    , d_intValue12(n.d_intValue12)
    , d_intValue13(n.d_intValue13)
    , d_intValue14(n.d_intValue14)
    , d_intValue15(n.d_intValue15)
    , d_messageIDIn(n.d_messageIDIn)
    , d_floatValueIn0(n.d_floatValueIn0)
    , d_floatValueIn1(n.d_floatValueIn1)
    , d_floatValueIn2(n.d_floatValueIn2)
    , d_floatValueIn3(n.d_floatValueIn3)
    , d_floatValueIn4(n.d_floatValueIn4)
    , d_floatValueIn5(n.d_floatValueIn5)
    , d_floatValueIn6(n.d_floatValueIn6)
    , d_floatValueIn7(n.d_floatValueIn7)
    , d_floatValueIn8(n.d_floatValueIn8)
    , d_floatValueIn9(n.d_floatValueIn9)
    , d_floatValueIn10(n.d_floatValueIn10)
    , d_floatValueIn11(n.d_floatValueIn11)
    , d_floatValueIn12(n.d_floatValueIn12)
    , d_floatValueIn13(n.d_floatValueIn13)
    , d_floatValueIn14(n.d_floatValueIn14)
    , d_floatValueIn15(n.d_floatValueIn15)
    , d_intValueIn0(n.d_intValueIn0)
    , d_intValueIn1(n.d_intValueIn1)
    , d_intValueIn2(n.d_intValueIn2)
    , d_intValueIn3(n.d_intValueIn3)
    , d_intValueIn4(n.d_intValueIn4)
    , d_intValueIn5(n.d_intValueIn5)
    , d_intValueIn6(n.d_intValueIn6)
    , d_intValueIn7(n.d_intValueIn7)
    , d_intValueIn8(n.d_intValueIn8)
    , d_intValueIn9(n.d_intValueIn9)
    , d_intValueIn10(n.d_intValueIn10)
    , d_intValueIn11(n.d_intValueIn11)
    , d_intValueIn12(n.d_intValueIn12)
    , d_intValueIn13(n.d_intValueIn13)
    , d_intValueIn14(n.d_intValueIn14)
    , d_intValueIn15(n.d_intValueIn15)
    , d_lastReceiveTime(n.d_lastReceiveTime)

{
}

/** Destructor.
*/
VrmlNodePorscheVFP::~VrmlNodePorscheVFP()
{
    // 	delete clientConn;
}

/** Return a clone. Calls Copy Constructor.
*/
VrmlNode *
VrmlNodePorscheVFP::cloneMe() const
{
    return new VrmlNodePorscheVFP(*this);
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

/** Print fields to stream.
*/
std::ostream &
VrmlNodePorscheVFP::printFields(std::ostream &os, int indent)
{
    PRINT_FIELD(targetIP);
    PRINT_FIELD(targetPort);
    // 	PRINT_FIELD(localPort);
    PRINT_FIELD(messageID);
    PRINT_FIELD(floatValue0);
    PRINT_FIELD(floatValue1);
    PRINT_FIELD(floatValue2);
    PRINT_FIELD(floatValue3);
    PRINT_FIELD(floatValue4);
    PRINT_FIELD(floatValue5);
    PRINT_FIELD(floatValue6);
    PRINT_FIELD(floatValue7);
    PRINT_FIELD(floatValue8);
    PRINT_FIELD(floatValue9);
    PRINT_FIELD(floatValue10);
    PRINT_FIELD(floatValue11);
    PRINT_FIELD(floatValue12);
    PRINT_FIELD(floatValue13);
    PRINT_FIELD(floatValue14);
    PRINT_FIELD(floatValue15);
    PRINT_FIELD(intValue0);
    PRINT_FIELD(intValue1);
    PRINT_FIELD(intValue2);
    PRINT_FIELD(intValue3);
    PRINT_FIELD(intValue4);
    PRINT_FIELD(intValue5);
    PRINT_FIELD(intValue6);
    PRINT_FIELD(intValue7);
    PRINT_FIELD(intValue8);
    PRINT_FIELD(intValue9);
    PRINT_FIELD(intValue10);
    PRINT_FIELD(intValue11);
    PRINT_FIELD(intValue12);
    PRINT_FIELD(intValue13);
    PRINT_FIELD(intValue14);
    PRINT_FIELD(intValue15);
    PRINT_FIELD(floatValueIn0);
    PRINT_FIELD(floatValueIn1);
    PRINT_FIELD(floatValueIn2);
    PRINT_FIELD(floatValueIn3);
    PRINT_FIELD(floatValueIn4);
    PRINT_FIELD(floatValueIn5);
    PRINT_FIELD(floatValueIn6);
    PRINT_FIELD(floatValueIn7);
    PRINT_FIELD(floatValueIn8);
    PRINT_FIELD(floatValueIn9);
    PRINT_FIELD(floatValueIn10);
    PRINT_FIELD(floatValueIn11);
    PRINT_FIELD(floatValueIn12);
    PRINT_FIELD(floatValueIn13);
    PRINT_FIELD(floatValueIn14);
    PRINT_FIELD(floatValueIn15);
    PRINT_FIELD(intValueIn0);
    PRINT_FIELD(intValueIn1);
    PRINT_FIELD(intValueIn2);
    PRINT_FIELD(intValueIn3);
    PRINT_FIELD(intValueIn4);
    PRINT_FIELD(intValueIn5);
    PRINT_FIELD(intValueIn6);
    PRINT_FIELD(intValueIn7);
    PRINT_FIELD(intValueIn8);
    PRINT_FIELD(intValueIn9);
    PRINT_FIELD(intValueIn10);
    PRINT_FIELD(intValueIn11);
    PRINT_FIELD(intValueIn12);
    PRINT_FIELD(intValueIn13);
    PRINT_FIELD(intValueIn14);
    PRINT_FIELD(intValueIn15);

    return os;
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

/** Set the value of one of the node fields.
*/
void
VrmlNodePorscheVFP::setField(const char *fieldName, const VrmlField &fieldValue)
{
    setModified();
    if
        TRY_FIELD(targetIP, SFString)
    else if
        TRY_FIELD(targetPort, SFInt)
    // 	else if TRY_FIELD(localPort, SFInt)
    else if
        TRY_FIELD(messageID, SFInt)
    else if
        TRY_FIELD(floatValue0, SFFloat)
    else if
        TRY_FIELD(floatValue1, SFFloat)
    else if
        TRY_FIELD(floatValue2, SFFloat)
    else if
        TRY_FIELD(floatValue3, SFFloat)
    else if
        TRY_FIELD(floatValue4, SFFloat)
    else if
        TRY_FIELD(floatValue5, SFFloat)
    else if
        TRY_FIELD(floatValue6, SFFloat)
    else if
        TRY_FIELD(floatValue7, SFFloat)
    else if
        TRY_FIELD(floatValue8, SFFloat)
    else if
        TRY_FIELD(floatValue9, SFFloat)
    else if
        TRY_FIELD(floatValue10, SFFloat)
    else if
        TRY_FIELD(floatValue11, SFFloat)
    else if
        TRY_FIELD(floatValue12, SFFloat)
    else if
        TRY_FIELD(floatValue13, SFFloat)
    else if
        TRY_FIELD(floatValue14, SFFloat)
    else if
        TRY_FIELD(floatValue15, SFFloat)
    else if
        TRY_FIELD(intValue0, SFInt)
    else if
        TRY_FIELD(intValue1, SFInt)
    else if
        TRY_FIELD(intValue2, SFInt)
    else if
        TRY_FIELD(intValue3, SFInt)
    else if
        TRY_FIELD(intValue4, SFInt)
    else if
        TRY_FIELD(intValue5, SFInt)
    else if
        TRY_FIELD(intValue6, SFInt)
    else if
        TRY_FIELD(intValue7, SFInt)
    else if
        TRY_FIELD(intValue8, SFInt)
    else if
        TRY_FIELD(intValue9, SFInt)
    else if
        TRY_FIELD(intValue10, SFInt)
    else if
        TRY_FIELD(intValue11, SFInt)
    else if
        TRY_FIELD(intValue12, SFInt)
    else if
        TRY_FIELD(intValue13, SFInt)
    else if
        TRY_FIELD(intValue14, SFInt)
    else if
        TRY_FIELD(intValue15, SFInt)
    else if
        TRY_FIELD(floatValueIn0, SFFloat)
    else if
        TRY_FIELD(floatValueIn1, SFFloat)
    else if
        TRY_FIELD(floatValueIn2, SFFloat)
    else if
        TRY_FIELD(floatValueIn3, SFFloat)
    else if
        TRY_FIELD(floatValueIn4, SFFloat)
    else if
        TRY_FIELD(floatValueIn5, SFFloat)
    else if
        TRY_FIELD(floatValueIn6, SFFloat)
    else if
        TRY_FIELD(floatValueIn7, SFFloat)
    else if
        TRY_FIELD(floatValueIn8, SFFloat)
    else if
        TRY_FIELD(floatValueIn9, SFFloat)
    else if
        TRY_FIELD(floatValueIn10, SFFloat)
    else if
        TRY_FIELD(floatValueIn11, SFFloat)
    else if
        TRY_FIELD(floatValueIn12, SFFloat)
    else if
        TRY_FIELD(floatValueIn13, SFFloat)
    else if
        TRY_FIELD(floatValueIn14, SFFloat)
    else if
        TRY_FIELD(floatValueIn15, SFFloat)
    else if
        TRY_FIELD(intValueIn0, SFInt)
    else if
        TRY_FIELD(intValueIn1, SFInt)
    else if
        TRY_FIELD(intValueIn2, SFInt)
    else if
        TRY_FIELD(intValueIn3, SFInt)
    else if
        TRY_FIELD(intValueIn4, SFInt)
    else if
        TRY_FIELD(intValueIn5, SFInt)
    else if
        TRY_FIELD(intValueIn6, SFInt)
    else if
        TRY_FIELD(intValueIn7, SFInt)
    else if
        TRY_FIELD(intValueIn8, SFInt)
    else if
        TRY_FIELD(intValueIn9, SFInt)
    else if
        TRY_FIELD(intValueIn10, SFInt)
    else if
        TRY_FIELD(intValueIn11, SFInt)
    else if
        TRY_FIELD(intValueIn12, SFInt)
    else if
        TRY_FIELD(intValueIn13, SFInt)
    else if
        TRY_FIELD(intValueIn14, SFInt)
    else if
        TRY_FIELD(intValueIn15, SFInt)

    else
        VrmlNodeChild::setField(fieldName, fieldValue); // base class
}

const VrmlField *
VrmlNodePorscheVFP::getField(const char *fieldName) const
{
    if (strcmp(fieldName, "targetIP") == 0)
        return &d_targetIP;
    else if (strcmp(fieldName, "targetPort") == 0)
        return &d_targetPort;
    // 	else if(strcmp(fieldName,"localPort")==0)
    // 		return &d_localPort;
    else if (strcmp(fieldName, "messageID") == 0)
        return &d_messageID;
    else if (strcmp(fieldName, "floatValue0") == 0)
        return &d_floatValue0;
    else if (strcmp(fieldName, "floatValue1") == 0)
        return &d_floatValue1;
    else if (strcmp(fieldName, "floatValue2") == 0)
        return &d_floatValue2;
    else if (strcmp(fieldName, "floatValue3") == 0)
        return &d_floatValue3;
    else if (strcmp(fieldName, "floatValue4") == 0)
        return &d_floatValue4;
    else if (strcmp(fieldName, "floatValue5") == 0)
        return &d_floatValue5;
    else if (strcmp(fieldName, "floatValue6") == 0)
        return &d_floatValue6;
    else if (strcmp(fieldName, "floatValue7") == 0)
        return &d_floatValue7;
    else if (strcmp(fieldName, "floatValue8") == 0)
        return &d_floatValue8;
    else if (strcmp(fieldName, "floatValue9") == 0)
        return &d_floatValue9;
    else if (strcmp(fieldName, "floatValue10") == 0)
        return &d_floatValue10;
    else if (strcmp(fieldName, "floatValue11") == 0)
        return &d_floatValue11;
    else if (strcmp(fieldName, "floatValue12") == 0)
        return &d_floatValue12;
    else if (strcmp(fieldName, "floatValue13") == 0)
        return &d_floatValue13;
    else if (strcmp(fieldName, "floatValue14") == 0)
        return &d_floatValue14;
    else if (strcmp(fieldName, "floatValue15") == 0)
        return &d_floatValue15;
    else if (strcmp(fieldName, "messageIDIn") == 0)
        return &d_messageID;
    else if (strcmp(fieldName, "intValue0") == 0)
        return &d_intValue0;
    else if (strcmp(fieldName, "intValue1") == 0)
        return &d_intValue1;
    else if (strcmp(fieldName, "intValue2") == 0)
        return &d_intValue2;
    else if (strcmp(fieldName, "intValue3") == 0)
        return &d_intValue3;
    else if (strcmp(fieldName, "intValue4") == 0)
        return &d_intValue4;
    else if (strcmp(fieldName, "intValue5") == 0)
        return &d_intValue5;
    else if (strcmp(fieldName, "intValue6") == 0)
        return &d_intValue6;
    else if (strcmp(fieldName, "intValue7") == 0)
        return &d_intValue7;
    else if (strcmp(fieldName, "intValue8") == 0)
        return &d_intValue8;
    else if (strcmp(fieldName, "intValue9") == 0)
        return &d_intValue9;
    else if (strcmp(fieldName, "intValue10") == 0)
        return &d_intValue10;
    else if (strcmp(fieldName, "intValue11") == 0)
        return &d_intValue11;
    else if (strcmp(fieldName, "intValue12") == 0)
        return &d_intValue12;
    else if (strcmp(fieldName, "intValue13") == 0)
        return &d_intValue13;
    else if (strcmp(fieldName, "intValue14") == 0)
        return &d_intValue14;
    else if (strcmp(fieldName, "intValue15") == 0)
        return &d_intValue15;
    else if (strcmp(fieldName, "floatValueIn0") == 0)
        return &d_floatValueIn0;
    else if (strcmp(fieldName, "floatValueIn1") == 0)
        return &d_floatValueIn1;
    else if (strcmp(fieldName, "floatValueIn2") == 0)
        return &d_floatValueIn2;
    else if (strcmp(fieldName, "floatValueIn3") == 0)
        return &d_floatValueIn3;
    else if (strcmp(fieldName, "floatValueIn4") == 0)
        return &d_floatValueIn4;
    else if (strcmp(fieldName, "floatValueIn5") == 0)
        return &d_floatValueIn5;
    else if (strcmp(fieldName, "floatValueIn6") == 0)
        return &d_floatValueIn6;
    else if (strcmp(fieldName, "floatValueIn7") == 0)
        return &d_floatValueIn7;
    else if (strcmp(fieldName, "floatValueIn8") == 0)
        return &d_floatValueIn8;
    else if (strcmp(fieldName, "floatValueIn9") == 0)
        return &d_floatValueIn9;
    else if (strcmp(fieldName, "floatValueIn10") == 0)
        return &d_floatValueIn10;
    else if (strcmp(fieldName, "floatValueIn11") == 0)
        return &d_floatValueIn11;
    else if (strcmp(fieldName, "floatValueIn12") == 0)
        return &d_floatValueIn12;
    else if (strcmp(fieldName, "floatValueIn13") == 0)
        return &d_floatValueIn13;
    else if (strcmp(fieldName, "floatValueIn14") == 0)
        return &d_floatValueIn14;
    else if (strcmp(fieldName, "floatValueIn15") == 0)
        return &d_floatValueIn15;
    else if (strcmp(fieldName, "intValueIn0") == 0)
        return &d_intValueIn0;
    else if (strcmp(fieldName, "intValueIn1") == 0)
        return &d_intValueIn1;
    else if (strcmp(fieldName, "intValueIn2") == 0)
        return &d_intValueIn2;
    else if (strcmp(fieldName, "intValueIn3") == 0)
        return &d_intValueIn3;
    else if (strcmp(fieldName, "intValueIn4") == 0)
        return &d_intValueIn4;
    else if (strcmp(fieldName, "intValueIn5") == 0)
        return &d_intValueIn5;
    else if (strcmp(fieldName, "intValueIn6") == 0)
        return &d_intValueIn6;
    else if (strcmp(fieldName, "intValueIn7") == 0)
        return &d_intValueIn7;
    else if (strcmp(fieldName, "intValueIn8") == 0)
        return &d_intValueIn8;
    else if (strcmp(fieldName, "intValueIn9") == 0)
        return &d_intValueIn9;
    else if (strcmp(fieldName, "intValueIn10") == 0)
        return &d_intValueIn10;
    else if (strcmp(fieldName, "intValueIn11") == 0)
        return &d_intValueIn11;
    else if (strcmp(fieldName, "intValueIn12") == 0)
        return &d_intValueIn12;
    else if (strcmp(fieldName, "intValueIn13") == 0)
        return &d_intValueIn13;
    else if (strcmp(fieldName, "intValueIn14") == 0)
        return &d_intValueIn14;
    else if (strcmp(fieldName, "intValueIn15") == 0)
        return &d_intValueIn15;

    else
        cerr << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
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
        outData.floatValues[0] = d_floatValue0.get();
        outData.floatValues[1] = d_floatValue1.get();
        outData.floatValues[2] = d_floatValue2.get();
        outData.floatValues[3] = d_floatValue3.get();
        outData.floatValues[4] = d_floatValue4.get();
        outData.floatValues[5] = d_floatValue5.get();
        outData.floatValues[6] = d_floatValue6.get();
        outData.floatValues[7] = d_floatValue7.get();
        outData.floatValues[8] = d_floatValue8.get();
        outData.floatValues[9] = d_floatValue9.get();
        outData.floatValues[10] = d_floatValue10.get();
        outData.floatValues[11] = d_floatValue11.get();
        outData.floatValues[12] = d_floatValue12.get();
        outData.floatValues[13] = d_floatValue13.get();
        outData.floatValues[14] = d_floatValue14.get();
        outData.floatValues[15] = d_floatValue15.get();
        outData.intValues[0] = d_intValue0.get();
        outData.intValues[1] = d_intValue1.get();
        outData.intValues[2] = d_intValue2.get();
        outData.intValues[3] = d_intValue3.get();
        outData.intValues[4] = d_intValue4.get();
        outData.intValues[5] = d_intValue5.get();
        outData.intValues[6] = d_intValue6.get();
        outData.intValues[7] = d_intValue7.get();
        outData.intValues[8] = d_intValue8.get();
        outData.intValues[9] = d_intValue9.get();
        outData.intValues[10] = d_intValue10.get();
        outData.intValues[11] = d_intValue11.get();
        outData.intValues[12] = d_intValue12.get();
        outData.intValues[13] = d_intValue13.get();
        outData.intValues[14] = d_intValue14.get();
        outData.intValues[15] = d_intValue15.get();

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

    d_floatValueIn0 = fromMaster.floatValues[0];
    d_floatValueIn1 = fromMaster.floatValues[1];
    d_floatValueIn2 = fromMaster.floatValues[2];
    d_floatValueIn3 = fromMaster.floatValues[3];
    d_floatValueIn4 = fromMaster.floatValues[4];
    d_floatValueIn5 = fromMaster.floatValues[5];
    d_floatValueIn6 = fromMaster.floatValues[6];
    d_floatValueIn7 = fromMaster.floatValues[7];
    d_floatValueIn8 = fromMaster.floatValues[8];
    d_floatValueIn9 = fromMaster.floatValues[9];
    d_floatValueIn10 = fromMaster.floatValues[10];
    d_floatValueIn11 = fromMaster.floatValues[11];
    d_floatValueIn12 = fromMaster.floatValues[12];
    d_floatValueIn13 = fromMaster.floatValues[13];
    d_floatValueIn14 = fromMaster.floatValues[14];
    d_floatValueIn15 = fromMaster.floatValues[15];

    d_intValueIn0 = fromMaster.intValues[0];
    d_intValueIn1 = fromMaster.intValues[1];
    d_intValueIn2 = fromMaster.intValues[2];
    d_intValueIn3 = fromMaster.intValues[3];
    d_intValueIn4 = fromMaster.intValues[4];
    d_intValueIn5 = fromMaster.intValues[5];
    d_intValueIn6 = fromMaster.intValues[6];
    d_intValueIn7 = fromMaster.intValues[7];
    d_intValueIn8 = fromMaster.intValues[8];
    d_intValueIn9 = fromMaster.intValues[9];
    d_intValueIn10 = fromMaster.intValues[10];
    d_intValueIn11 = fromMaster.intValues[11];
    d_intValueIn12 = fromMaster.intValues[12];
    d_intValueIn13 = fromMaster.intValues[13];
    d_intValueIn14 = fromMaster.intValues[14];
    d_intValueIn15 = fromMaster.intValues[15];

    // Events //
    //
    double timeStamp = System::the->time();

    eventOut(timeStamp, "messageIDIn", d_messageIDIn);

    eventOut(timeStamp, "floatValueIn0", d_floatValueIn0);
    eventOut(timeStamp, "floatValueIn1", d_floatValueIn1);
    eventOut(timeStamp, "floatValueIn2", d_floatValueIn2);
    eventOut(timeStamp, "floatValueIn3", d_floatValueIn3);
    eventOut(timeStamp, "floatValueIn4", d_floatValueIn4);
    eventOut(timeStamp, "floatValueIn5", d_floatValueIn5);
    eventOut(timeStamp, "floatValueIn6", d_floatValueIn6);
    eventOut(timeStamp, "floatValueIn7", d_floatValueIn7);
    eventOut(timeStamp, "floatValueIn8", d_floatValueIn8);
    eventOut(timeStamp, "floatValueIn9", d_floatValueIn9);
    eventOut(timeStamp, "floatValueIn10", d_floatValueIn10);
    eventOut(timeStamp, "floatValueIn11", d_floatValueIn11);
    eventOut(timeStamp, "floatValueIn12", d_floatValueIn12);
    eventOut(timeStamp, "floatValueIn13", d_floatValueIn13);
    eventOut(timeStamp, "floatValueIn14", d_floatValueIn14);
    eventOut(timeStamp, "floatValueIn15", d_floatValueIn15);

    eventOut(timeStamp, "intValueIn0", d_intValueIn0);
    eventOut(timeStamp, "intValueIn1", d_intValueIn1);
    eventOut(timeStamp, "intValueIn2", d_intValueIn2);
    eventOut(timeStamp, "intValueIn3", d_intValueIn3);
    eventOut(timeStamp, "intValueIn4", d_intValueIn4);
    eventOut(timeStamp, "intValueIn5", d_intValueIn5);
    eventOut(timeStamp, "intValueIn6", d_intValueIn6);
    eventOut(timeStamp, "intValueIn7", d_intValueIn7);
    eventOut(timeStamp, "intValueIn8", d_intValueIn8);
    eventOut(timeStamp, "intValueIn9", d_intValueIn9);
    eventOut(timeStamp, "intValueIn10", d_intValueIn10);
    eventOut(timeStamp, "intValueIn11", d_intValueIn11);
    eventOut(timeStamp, "intValueIn12", d_intValueIn12);
    eventOut(timeStamp, "intValueIn13", d_intValueIn13);
    eventOut(timeStamp, "intValueIn14", d_intValueIn14);
    eventOut(timeStamp, "intValueIn15", d_intValueIn15);

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
