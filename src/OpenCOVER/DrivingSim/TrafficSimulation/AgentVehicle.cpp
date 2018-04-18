/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "AgentVehicle.h"
#include <VehicleUtil/RoadSystem/Junction.h>
#include <VehicleUtil/RoadSystem/Lane.h>
#include "TrafficSimulation.h"
#include <algorithm>
#include <math.h>
#include <float.h> //für die #INF-Abfrage (DBL_MAX und DBL_MIN)

AgentVehicle::AgentVehicle(AgentVehicle *veh, std::string name, const VehicleParameters &vp, Road *r, double startu, int startLane, double startVel, int startDir)
    : Vehicle(name)
    , vehPars(vp)
    , vehState()
    , currentLane(startLane)
    , u(startu)
    , signalBarrierTimer(0.0)
    , geometry(NULL)
    , vel(startVel)
    , ms2kmh(3.6)
    , kv(5.0)
    , dampingV(5.0)
    , Tv(5)
    , hdg(0.0)
    , dHdg(0.0)
    , Khdg(2.0)
    , Dhdg(20.0)
    , Kv(0.2)
    , Dv(2.0)
    , velt(startVel)
    , minTransitionListExtent(1000)
    , boundRadius(0.0)
    , junctionWaitTime(0.5)
    , junctionWaitTriggerDist(2)
    , sowedDetermineNextRoadVehicleAction(false)
    , repeatRoute(false)
    , timer(0.0)
    , vehType(ObstacleData::OBSTACLE_SPORTSCAR)
    , currentCrosswalk(NULL)
    , crossId(Crosswalk::DONOTENTER)
    , crossing(false)
    , crossPollTimer(0.0)
    , canBePassed(true)

{
    // Neu Andreas 27-11-2012
    currentSpeedLimit = 50.0;
    velDev = 0.0;
    v = 0.0;
    s = 0.0;
    du = 0.0;
    dv = 0.0;
    lcTime = 0.0;
    hdgAlt = 0.0;
    lcV = 0.0;
    velo = 0.0;
    currentRoadType = Road::UNKNOWN;
    Pool *currentPool = Carpool::Instance()->getPoolById(name);
    if (currentPool)
    {
        // Höchstgeschwindigkeit aus Carpool
        vehPars.dUtarget = currentPool->getStartVelocity();
    }
    else
    {
        // Höchstgeschwindigkeit aus Fiddlesource
        vehPars.dUtarget = startVel;
    }

    if (r)
    {
        s = 0;
        du = startVel;
        dv = 0;
        lcTime = timer;
        lcV = 0; //die Position v auf der Straße, bevor man den Spurwechsel einleitet
        velo = du; //Hilfsvariable für die Berechnung der Geschwindigkeit

        roadTransitionList.push_back(RoadTransition(r, startDir));
        currentTransition = roadTransitionList.begin();
        extendSignalBarrierList(*currentTransition);
        currentSection = r->getLaneSection(startu);

        // Neu Andreas 27-11-2012
        currentSpeedLimit = currentTransition->road->getSpeedLimit(u, currentLane);
        velDev = (startVel - currentSpeedLimit) / currentSpeedLimit;

        Vector2D laneCenter = currentSection->getLaneCenter(startLane, startu);
        v = laneCenter[0];
        //hdg =  M_PI*((currentTransition->direction - 1)/2) + laneCenter[1];
        hdg = M_PI * ((1.0 - currentTransition->direction) * 0.5);

        vehicleTransform = r->getRoadTransform(startu, laneCenter[0]);

        //currentRoadId = currentTransition->road->getId();		//löschen!
        coordDeque.push_back(std::pair<double, double>(u, v));
        hdgAlt = hdg;
    }

    CarGeometry *geo = new CarGeometry(veh->getCarGeometry(), name);
    if (geo)
    {
        geometry = geo;
        boundRadius = geo->getBoundingCircleRadius();
        geometry->setLODrange(vehPars.rangeLOD);
        geometry->setTransform(vehicleTransform, hdg);
        junctionWaitTriggerDist += boundRadius + vehPars.deltaSmin;
    }

    init();
}

AgentVehicle::AgentVehicle(std::string name, CarGeometry *geo, const VehicleParameters &vp, Road *r, double startu, int startLane, double startVel, int startDir)
    : Vehicle(name)
    , vehPars(vp)
    , vehState()
    , currentLane(startLane)
    , u(startu)
    , signalBarrierTimer(0.0)
    , geometry(NULL)
    , vel(startVel)
    , ms2kmh(3.6)
    , kv(5.0)
    , dampingV(5.0)
    , Tv(5)
    , hdg(0.0)
    , dHdg(0.0)
    , Khdg(2.0)
    , Dhdg(20.0)
    , Kv(0.2)
    , Dv(2.0)
    , velt(startVel)
    , minTransitionListExtent(1000)
    , boundRadius(0.0)
    , junctionWaitTime(0.5)
    , junctionWaitTriggerDist(2)
    , sowedDetermineNextRoadVehicleAction(false)
    , repeatRoute(false)
    , timer(0.0)
    , vehType(ObstacleData::OBSTACLE_SPORTSCAR)
    , currentCrosswalk(NULL)
    , crossId(Crosswalk::DONOTENTER)
    , crossing(false)
    , crossPollTimer(0.0)
    , canBePassed(true)

{
    vehPars.dUtarget = startVel;
    currentRoadType = Road::UNKNOWN;
    currentSpeedLimit = 50.0;
    velDev = 0.0;
    v = 0.0;
    s = 0.0;
    du = 0.0;
    dv = 0.0;
    hdgAlt = 0.0;
    lcTime = 0.0;
    lcV = 0.0;
    velo = 0.0;
    if (r)
    {
        s = 0;
        du = startVel;
        dv = 0;
        lcTime = timer;

        lcV = 0; //die Position v auf der Straße, bevor man den Spurwechsel einleitet
        velo = du; //Hilfsvariable für die Berechnung der Geschwindigkeit

        roadTransitionList.push_back(RoadTransition(r, startDir));
        currentTransition = roadTransitionList.begin();
        extendSignalBarrierList(*currentTransition);

        currentSection = r->getLaneSection(startu);

        // Neu Andreas 27-11-2012
        currentSpeedLimit = currentTransition->road->getSpeedLimit(u, currentLane);
        velDev = (startVel - currentSpeedLimit) / 100;

        Vector2D laneCenter = currentSection->getLaneCenter(startLane, startu);
        v = laneCenter[0];
        //hdg =  M_PI*((currentTransition->direction - 1)/2) + laneCenter[1];
        hdg = M_PI * ((1 - currentTransition->direction) / 2);

        vehicleTransform = r->getRoadTransform(startu, laneCenter[0]);

        //currentRoadId = currentTransition->road->getId();			//löschen!
        coordDeque.push_back(std::pair<double, double>(u, v));
        hdgAlt = hdg;
    }

    if (geo)
    {
        geometry = geo;
        boundRadius = geo->getBoundingCircleRadius();
        geometry->setLODrange(vehPars.rangeLOD);
        geometry->setTransform(vehicleTransform, hdg);
        junctionWaitTriggerDist += boundRadius + vehPars.deltaSmin;
    }

    init();
}

void AgentVehicle::init()
{
    drivableLaneTypeSet.insert(Lane::DRIVING); //Fahrbahntypen, auf denen der Fahrzeugagent fahren darf
    drivableLaneTypeSet.insert(Lane::MWYEXIT);
    drivableLaneTypeSet.insert(Lane::MWYENTRY);
}

AgentVehicle::~AgentVehicle()
{
    // If the vehicle has reserved a crosswalk...
    if (currentCrosswalk)
        currentCrosswalk->exitVehicle(crossId, opencover::cover->frameTime());

    delete geometry;
}

void AgentVehicle::move(double dt)
{

    // Timer //
    //
    timer += dt;

    // Follow given route //
    //
    if (!routeTransitionList.empty())
    {
        planRoute();
        std::cout << "Vehicle " << name << ": Road transition list";
        if (repeatRoute)
            std::cout << " (repeat)";
        std::cout << ":";
        for (RoadTransitionList::iterator transIt = roadTransitionList.begin(); transIt != roadTransitionList.end(); ++transIt)
        {
            std::cout << " " << transIt->road->getId();
            if (transIt->junction)
                std::cout << "-j:" << transIt->junction->getId();
        }
        std::cout << std::endl;
    }
    else if (repeatRoute && !roadTransitionList.empty() && (--roadTransitionList.end() == currentTransition))
    {
        routeTransitionList.insert(routeTransitionList.end(), routeTransitionListBackup.begin(), routeTransitionListBackup.end());
    }
    // Find your own way //
    //
    else if (!sowedDetermineNextRoadVehicleAction && roadTransitionList.size()>0 && (--roadTransitionList.end() == currentTransition) && !repeatRoute)
    {
        //std::cout << "Inserting determine next road vehicle action at s: " << s << std::endl;
        vehiclePositionActionMap.insert(std::pair<double, VehicleAction *>(s, new DetermineNextRoadVehicleAction())); // s are the total m so far (not u)
        sowedDetermineNextRoadVehicleAction = true;
        executeActionMap();
        //std::cout << "Road transition list:";
        //for(RoadTransitionList::iterator transIt = roadTransitionList.begin(); transIt!=roadTransitionList.end(); ++transIt) {
        //   std::cout << " " << transIt->road->getId();
        //}
        //std::cout << std::endl;
    }

    // Just in case //
    //
    if (currentLane == Lane::NOLANE)
    {
        executeActionMap();
        return;
    }

    // Limiting speed //
    //
    {
        // reasons: max speed of vehicle, curvature
        //Vector3D vehVecRel = vehicleTransform.v() - currentTransition->road->getCenterLinePoint(u);
        //Vector3D vehVecRelUnit = vehVecRel * (1.0/(vehVecRel.length()));
        //Vector3D normalVectorCurve = currentTransition->road->getNormalVector(u);
        //Vector3D normalVectorOffset =  normalVectorCurve * (1/(1 - normalVectorCurve.dot(vehVecRel)));
        //double veltcQuad = (vehPars.accCrossmax + vehVecRelUnit.dot(Vector3D(0,0,-9.81))) / (vehVecRelUnit.dot(normalVectorOffset));
        //double veltc = sqrt(fabs( veltcQuad ) );
        //std::cout << "Vehicle " << name << ": veltc: " << veltc << ", veltcQuad: " << veltcQuad << std::endl;
        //std::cout << "vehVecRelUnit: " << vehVecRelUnit;

        // Max speed due to curvature //
        //
        double actCurv = fabs(currentTransition->road->getCurvature(u));
        double actElev = currentTransition->road->getSuperelevation(u, v);

        // Neu Andreas 27-11-2012: SpeedLimit
        currentSpeedLimit = currentTransition->road->getSpeedLimit(u, currentLane);
        // Add Velocity Deviance to SpeedLimit
        if (velDev > 0)
        {
            currentSpeedLimit += velDev * currentSpeedLimit;
        }
        if (currentSpeedLimit > vehPars.dUtarget)
        {
            currentSpeedLimit = vehPars.dUtarget;
        }

        double veltc = currentSpeedLimit;
        if (actCurv != 0.0)
            veltc = sqrt(vehPars.accCrossmax / (actCurv * actCurv * (1 / actCurv + v) * cos(actElev)));
        //double veltc = sqrt(vehPars.accCrossmax/(actCurv*actCurv*(1/actCurv + v)*cos(actElev)));
        //std::cout << "veltc: " << veltc << std::endl;
        if (veltc != veltc) // not the same if term in sqrt is negative
        {
            veltc = currentSpeedLimit;
        }
        /*double velc = sqrt(fabs(vehPars.accCrossmax/currentTransition->road->getCurvature(u)));
      if(velc==velc) {
         velt = std::min(velc, vehPars.dUtarget);
      }
      else {
         velt = vehPars.dUtarget;
      }*/
        //std::cout << "vel: " << du << ", velt: " << velt << ", velc: " << velc << ", velw: " << velw << std::endl;

        /*
	  velt...		Sollgeschwindigkeit/ Wunschgeschwindigkeit
	  veltc...		maximale Geschwindigkeit des Fahrzeugagenten, die er an der Position u fahren kann, damit die auf
					das Fahrzeug wirkende Querkraft nicht zu groß wird
	  dUtarget...	max. Geschwindigkeit, die der Fahrzeugagent zu fahren bereit ist
	  */
        velt = std::min(veltc, vehPars.dUtarget);

        // Max speed due to RoadType and Speedlimit//
        velt = std::min(veltc, currentSpeedLimit);
        //std::cout<<"Sollgeschwindigkeit"<<currentSpeedLimit*ms2kmh<<"km/h"<<std::endl;
        // ...
    }
    //return; T 3.0

    double lastU = u;

    ObstacleRelation obsRel = locateVehicle(currentLane, 1);
    double laneEnd = locateLaneEnd(currentLane);

    //signal barrier
    if (!signalBarrierList.empty() && (*currentTransition) == signalBarrierList.front().first)
    {
        RoadSignal *signal = signalBarrierList.front().second;
        double ds = currentTransition->direction * (signal->getS() - u);
        //std::cout << "Signal barrier eminent..., ds: " << ds << std::endl;
        if (ds < 3.0)
        {
            signalBarrierTimer += dt;
            //std::cout << "Halt at signal barrier..." << std::endl;
            if (signalBarrierTimer > signal->getValue())
            {
                signalBarrierList.pop_front();
                signalBarrierTimer = 0.0;
                //std::cout << "Releasing signal barrier!" << std::endl;
            }
            else
            {
                laneEnd = ds;
            }
        }
        else
        {
            laneEnd = ds;
        }
    }
    else
    {
        /*
	  Beschleunigungsanteil auf freier Strecke
	  (bewirkt die Beschleunigung des Fahrzeugagenten auf die Sollgeschwindigkeit velt)
	  */
        signalBarrierTimer = 0.0;
    }

    //return; T 14.0

    // Dynamics //
    //
    double acc = 0.0;
    /* neue Version von Florian
   // Runge-Kutta
   {   //Verskriklike metode om die bewegingsvergelyking op te los!!!
      //acc += brakeDec;
      //brakeDec = 0.0;

      double deltaU_0 = 0.0;
      double deltaDu_0 = 0.0;
      double acc_0 = getAcceleration(obsRel, laneEnd, this->du, deltaU_0, deltaDu_0);

      double deltaU_A = 0.5*dt*(this->du + deltaDu_0);
      double deltaDu_A = 0.5*dt*acc_0;
      double acc_A = getAcceleration(obsRel, laneEnd, this->du, deltaU_A, deltaDu_A);

      double deltaU_B = 0.5*dt*(this->du + deltaDu_A);
      double deltaDu_B = 0.5*dt*acc_A;
      double acc_B = getAcceleration(obsRel, laneEnd, this->du, deltaU_B, deltaDu_B);

      double deltaU_C = dt*(this->du + deltaDu_B);
      double deltaDu_C = dt*acc_B;
      double acc_C = getAcceleration(obsRel, laneEnd, this->du, deltaU_C, deltaDu_C);

      double deltaU = dt*(this->du + 1.0/6.0*(deltaDu_0 + 2.0*(deltaDu_A+deltaDu_B) + deltaDu_C));
      double deltaDu = dt*1.0/6.0*(acc_0 + 2.0*(acc_A + acc_B) + acc_C);

      vel += deltaDu;
      if(vel<0) { //Dis nie so goed nie om die !(vel<0) beperking te verdwing
         vel = 0;
      }
   }
   */
    // neue Version von Porsche

    double vehDiffDu = -obsRel.diffDu;
    double vehDiffU = fabs(obsRel.diffU);

    if (!obsRel.noOR() && vehDiffU != 0)
    {
        /*
	  differentielle Bewegungsgleichung des Intelligent-Driver Modell von Treiber und Helbig bei vorausfahrendem Fahrzeug

	  acc...		Beschleunigung
	  accMax...		maximal mögliche Beschleunigung des Fahrzeugagenten
	  du...			Geschwindigkeit des Agenten
	  velt...		Sollgeschwindigkeit
	  approachFactor...bestimmt die Aggresivität der Annäherung an die Sollgeschwindigkeit
	  etd...		der effektive Wunschabstand
	  vehDiffDu...	Geschwindigkeit des Hindernisses relativ zum Fahrzeugagent
	  vehDiffU...	Abstand zum Hinderniss
	  */
        acc = vehPars.accMax * (1 - pow((this->du / velt), vehPars.approachFactor) - pow(etd(this->du, vehDiffDu) / vehDiffU, 2));
    }
    else
    {
        /*
	  Beschleunigungsanteil auf freier Strecke
	  (bewirkt die Beschleunigung des Fahrzeugagenten auf die Sollgeschwindigkeit velt)
	  */
        acc = vehPars.accMax * (1 - pow((fabs(this->du) / velt), vehPars.approachFactor));
    }

    //Berechnung der Beschleunigung bei Spurende

    double accInt = 0.0;
    if (laneEnd < 1e10)
    {
        accInt = vehPars.accMax * (pow(etd(this->du, this->du) / laneEnd, 2.0));
    }
    double accEol = vehPars.accMax * (1 - pow((this->du / velt), vehPars.approachFactor)) - accInt;

    if (accEol < acc)
    {
        acc = accEol;
    }

    //return; T 14.0

    // Dynamics //
    //
    // Euler vorwaerts
    //   {   //Verskriklike metode om die bewegingsvergelyking op te los!!!
    //      //acc += brakeDec;
    //      //brakeDec = 0.0;
    //
    //      vel += acc*dt;		//vel... Geschwindigkeit des Agenten bei Beschleunigung acc
    //      if(vel<0 && acc<0) { //Dis nie so goed nie om die !(vel<0) beperking te verdwing
    //         acc = 0;
    //         vel = 0;
    //      }
    //
    //      Vector2D laneCenter = currentSection->getLaneCenter(currentLane, u);
    //
    //      double vdiff = laneCenter[0] - v;
    //
    //	  /* alter Code vom Florian - war bereits auskommentiert **/
    //      //double ddHdg = currentTransition->direction*(vdiff*Kv*vel  - dHdg*Dhdg) + (M_PI*((1-currentTransition->direction)/2)-hdg)*Khdg - dv*Dv;
    //      //dHdg += ddHdg*dt;
    //      //hdg += dHdg*dt;
    //      //double dw = currentTransition->direction*vel*cos(hdg);
    //      //dv = currentTransition->direction*vel*sin(hdg);
    //
    //	  dv += (vdiff*kv*std::min(1.0, pow(timer-lcTime,2.0)) - dampingV*dv)*dt;
    //	  double dw = sqrt(vel*vel - dv*dv);	//Tangentialgeschwindigkeit zur Straße
    //      if(dw!=dw) {							//für den Fall NaN, dass in der Wurzel was negatives rauskommt
    //         dw=0;
    //		 dv = 0;
    //      }
    //
    //      du = 1/(-currentTransition->road->getCurvature(u) * v * cos(currentTransition->road->getSuperelevation(u,v)) + 1) * dw;
    //
    //	  //wenn die Geschw. <= 5 km/h, darf sich der Agent nicht quer zur Straße bewegen
    //	  if(du <= 1.4){
    //			  dv = 0;
    //		}
    //
    //	  //wenn der Abstand zum Hinderniss kleiner ist als der Mindestabstand, soll der Agent stehen bleiben
    //	  if(!obsRel.noOR() && obsRel.diffU <= vehPars.deltaSmin && obsRel.diffU != 0){
    //			 du = 0;
    //			 dv = 0;
    //		}
    //
    //	  u += currentTransition->direction*du*dt;
    //      v += dv*dt;
    //      s += du*dt;
    //
    //	/* --------- Heading neu von Olga -----------
    //	u- und v-Koordinaten der letzten ca. 2.5 Meter werden in einer Deque gespeichert und
    //	mit deren Hilfe das Heading ausgerechnet */
    //
    //	  // Change to next road //
    //	  double roadLength = currentTransition->road->getLength();
    //	  bool roadChange = false;
    //
    //	  //beim Uebergang von einer Straße auf eine andere, soll das vorherige Heading beibehalten werden, um ein Zittern zu verhindern
    //	  if((u > roadLength) || (u < 0.0)){
    //		   roadChange = true;
    //		   hdg = hdgAlt;
    //	   }
    //
    //	  coordDeque.push_back(std::pair<double,double>(u, v));	 //u- und v-Koordinaten in der Koordinaten-Deque speichern
    //	  while(fabs(coordDeque.front().first - coordDeque.back().first) > 2.5){
    //		  coordDeque.pop_front();
    //	  }
    //
    //	  if(roadChange == false && fabs(coordDeque.front().first - coordDeque.back().first) > 0.15){
    //		  hdg = M_PI*((currentTransition->direction - 1)/2) + atan((coordDeque.front().second - coordDeque.back().second) / (coordDeque.front().first - coordDeque.back().first));
    //		  if (hdg!=hdg) {
    //			  hdg = M_PI*((currentTransition->direction - 1)/2);
    //			  std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
    //		  }
    //	  }
    //
    //	  hdgAlt = hdg;
    //// --------- Heading neu Ende -----------
    //
    //      if(s!=s || v!=v || hdg!=hdg) {
    //         std::cout << "Vehicle " << name << ": s: " << s <<  ", dv: " << dv << ", dw: " << dw << ", du: " << du << ", hdg: " << hdg << ", u: " << u << ", v: " << v << std::endl;
    //         TrafficSimulation::instance()->haltSimulation();
    //      }
    //   }

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Spurwechselvorgang mit Sinus
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    {
        /** Dynamics NEU
	    Spurwechsel, Longitudinal- und Transversalgeschwindigkeit mit Sinus-Ansatz, Orientierung des Fahrzeugs */

        Vector2D laneCenter = currentSection->getLaneCenter(currentLane, u); //die Mitte der currentLane an der Position u
        double deltaT = 3; //Zeit in Sekunden, um den Spurwechsel durchzuführen
        double tRad = 0; //von der Zeit (timer) abhängiger Parameter in Bogenmaß
        //er soll auf der rechten Straßenseite von 0 in Richtung PI bzw. auf der linken Straßenseite von 0 in Richtung -PI verlaufen
        if (acc != acc || acc > vehPars.accMax || acc < -DBL_MAX)
        {
            acc = 0;
        }

        velo += acc * dt;

        if (velo > vehPars.dUtarget)
        {
            velo = vehPars.dUtarget;
        }

        /** Spurwechsel */
        if (fabs(v - laneCenter[0]) > 0.2)
        {
            double tRad = fmod((currentTransition->direction * (timer - lcTime) * M_PI / deltaT), M_PI);
            dv = ((laneCenter[0] - lcV) / 2) * cos(tRad - M_PI / 2) * (currentTransition->direction);

            if (dv * dv > velo * velo)
            {
                du = 0;
                velo = 0;
                dv = 0;
            }
            else
            {
                du = sqrt(velo * velo - dv * dv);
                du = 1 / (-currentTransition->road->getCurvature(u) * v * cos(currentTransition->road->getSuperelevation(u, v)) + 1) * du;
            }
        }

        else
        { /** Geradeaus-Fahren */
            dv = 0;
            du = 1 / (-currentTransition->road->getCurvature(u) * v * cos(currentTransition->road->getSuperelevation(u, v)) + 1) * velo;
            if (du < 0)
            {
                du = 0;
                velo = 0;
            }
        }

        if (du <= 1.4)
        {
            dv = 0;
        }
        //Mindestabstand erreicht -> nicht weiterfahren, stehen bleiben
        if (!obsRel.noOR() && vehDiffU <= vehPars.deltaSmin)
        {
            du = 0;
            velo = 0;
            dv = 0;
        }

        u += currentTransition->direction * du * dt;
        
        v += dv * dt;
        s += du * dt;

        /* --------- Heading neu -----------
	u- und v-Koordinaten der letzten ca. 2.5 Meter werden in einer Deque gespeichert und
	mit deren Hilfe das Heading ausgerechnet */

        // Change to next road //
        double roadLength = currentTransition->road->getLength();
        bool roadChange = false;

        //beim Uebergang von einer Straße auf eine andere, soll das vorherige Heading beibehalten werden, um ein Zittern zu verhindern
        if ((u > roadLength) || (u < 0.0))
        {
            roadChange = true;
            hdg = hdgAlt;
        }

        coordDeque.push_back(std::pair<double, double>(u, v)); //u- und v-Koordinaten in der Koordinaten-Deque speichern
        while (fabs(coordDeque.front().first - coordDeque.back().first) > 2.5)
        {
            coordDeque.pop_front();
        }

        // Neu Andreas: Hier entstand ein Bug bei Kreuzungen, Hack
        if (roadChange == false && fabs(coordDeque.front().first - coordDeque.back().first) > 0.15 && !(currentTransition->junction))
        {
            hdg = M_PI * ((currentTransition->direction - 1) / 2) + atan((coordDeque.front().second - coordDeque.back().second) / (coordDeque.front().first - coordDeque.back().first));
            if (hdg != hdg)
            {
                hdg = M_PI * ((currentTransition->direction - 1) / 2);
            }
        }

        hdgAlt = hdg;
        // --------- Heading neu -----------

        if (s != s || v != v || hdg != hdg)
        {
            std::cout << "Vehicle " << name << "timer: " << timer << ": s: " << s << ", du: " << du << ", acc: " << acc << ", direction: "
                      << currentTransition->direction << ", velo: " << velo << ", dv: " << dv << ", hdg: " << hdg << ", u: " << u << ", v: " << v << std::endl;
            std::cout << "vehDiffU: " << vehDiffU << ", vehDiffDu:" << vehDiffDu << ", vehicle: " << &Vehicle::getVehicleID << std::endl;
            TrafficSimulation::instance()->haltSimulation();
        }
    }

    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ Sinus-Ende
    //++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    //return; T 13.0

    //Action Map //
    //
    executeActionMap();

    //return; T 14.0

    double roadLength = currentTransition->road->getLength();

    // Change to next road //
    //
    if ((u > roadLength) || (u < 0.0))
    {
        if (u > roadLength)
        {
            u -= roadLength;

            if (u > roadLength)
            { // make sure we are not our of bounds
                u = roadLength;
            }
            if (u < 0.0)
            {
                u = 0.0;
            }
            v -= currentSection->getDistanceToLane(roadLength, currentLane);
        }
        else if (u < 0.0)
        {
            u = -u;

            if (u > roadLength)
            { // make sure we are not our of bounds
                u = roadLength;
            }
            if (u < 0.0)
            {
                u = 0.0;
            }
            v -= currentSection->getDistanceToLane(0.0, currentLane);
        }

        std::list<RoadTransition>::iterator nextTransition = currentTransition;
        ++nextTransition;
        /*if(repeatRoute && (nextTransition == roadTransitionList.end())) {
         //nextTransition = roadTransitionList.begin();
         //std::cout << "reseting transition list iterator!" << std::endl;
      }*/
        if (nextTransition != roadTransitionList.end())
        {
            u = (nextTransition->direction > 0) ? u : (nextTransition->road->getLength() - u);
            if (u < 0)
                u = 0.0;
            lastU = (nextTransition->direction > 0) ? 0.0 : nextTransition->road->getLength();
            if (lastU < 0)
                lastU = 0.0;

            //Junction* junction = nextTransition->road->getJunction();
            Junction *junction = nextTransition->junction;
            if (junction)
            {
                int newLane = junction->getConnectingLane(currentTransition->road, nextTransition->road, currentLane);
                if (newLane != Lane::NOLANE)
                {
                    currentLane = newLane;
                }
            }
            else
            {
                currentLane = (currentTransition->direction > 0) ? currentSection->getLaneSuccessor(currentLane) : currentSection->getLanePredecessor(currentLane);
            }
            currentSection = nextTransition->road->getLaneSection(u);

            if (currentLane == Lane::NOLANE)
            {
                vel = 0;
                return;
            }

            v *= currentTransition->direction * nextTransition->direction;
            v += currentSection->getDistanceToLane(u, currentLane);
            hdg += M_PI * ((currentTransition->direction * nextTransition->direction - 1) / 2);

            TrafficSimulation::instance()->getVehicleManager()->changeRoad(this, currentTransition->road, nextTransition->road, nextTransition->direction);
            currentTransition = nextTransition;
            extendSignalBarrierList(*currentTransition);
            //std::cout << "signal barrier list size: " << signalBarrierList.size() << std::endl;
            //std::cout << "Changing to road " << currentTransition->road->getId() << ", road transition list size: " << roadTransitionList.size() << std::endl;
        }
    }

    // Change to next LaneSection //
    //
    else
    {
        LaneSection *newSection = currentTransition->road->getLaneSection(u);
        if ((currentSection != newSection))
        {
            int oldLane = currentLane;
            currentLane = (currentTransition->direction > 0) ? currentSection->getLaneSuccessor(currentLane) : currentSection->getLanePredecessor(currentLane);
            currentSection = newSection;
            if (currentLane == Lane::NOLANE)
            {
                vel = 0;
                currentLane = oldLane;
                return;
            }
        }
    }

    // UPDATE VEHICLE STATE //
    //
    // Dynamics
    vehState.u = u;
    vehState.v = v;
    vehState.du = du;
    vehState.dv = dv;
    vehState.ddu = acc;
    vehState.hdg = hdg;
    vehState.dir = currentTransition->direction;

    // Lane Change || Junction => Indicators
    // Junction has higher priority.
    if (vehState.junctionState == VehicleState::JUNCTION_RIGHT)
    {
        if (!vehState.indicatorRight) // start laneChange
        {
            vehState.indicatorTstart = timer;
            vehState.indicatorRight = true;
            vehState.indicatorLeft = false;
        }
    }
    else if (vehState.junctionState == VehicleState::JUNCTION_LEFT)
    {
        if (!vehState.indicatorLeft) // start laneChange
        {
            vehState.indicatorTstart = timer;
            vehState.indicatorRight = false;
            vehState.indicatorLeft = true;
        }
    }
    else
    {
#define INDICATOR_MIN_V 0.1
        if (fabs(vehState.dv) > INDICATOR_MIN_V && vehState.dv * vehState.dir > 0.0)
        {
            // (dv and dir: both negative or both positive => left)
            if (!vehState.indicatorLeft) // start laneChange
            {
                vehState.indicatorTstart = timer;
                vehState.indicatorRight = false;
                vehState.indicatorLeft = true;
            }
        }
        else if (fabs(vehState.dv) > INDICATOR_MIN_V && vehState.dv * vehState.dir < 0.0)
        {
            if (!vehState.indicatorRight) // start laneChange
            {
                vehState.indicatorTstart = timer;
                vehState.indicatorRight = true;
                vehState.indicatorLeft = false;
            }
        }
        else
        {
            vehState.indicatorRight = false;
            vehState.indicatorLeft = false;
        }
    }

    // Check for upcoming crosswalk
    checkForCrosswalk(dt);

    // UPDATE GEOMETRY //
    //
    vehicleTransform = currentTransition->road->getRoadTransform(u, v);
    geometry->setTransform(vehicleTransform, hdg);
    geometry->updateCarParts(timer, dt, vehState);

    //alter Code von Florian Seybold - war bereits auskommentiert
    /*if(name=="schmotzcar") {
      std::vector<ObstacleRelation> relVec = getNeighboursOnLane(currentLane);
      ObstacleRelation thisRel(this, 0.0, 0.0);
      relVec[1] = thisRel;
      std::vector<double> accVec = computeVehicleAccelerations(relVec);
      std::cout << "Vehicle Accelerations:" << std::endl;
      for(int accIt = 0; accIt < accVec.size(); ++accIt) {
         std::cout << "\t Pos " << accIt << ": veh: " << relVec[accIt].vehicle << ", diffU: " << relVec[accIt].diffU << ", diffDu: " << relVec[accIt].diffDu << ", accVec: " << accVec[accIt] << std::endl;
      }
   }*/
    //std::cout << "Vehicle " << name << ": pos: " << u << ", lane: " << currentLane << std::endl;
} // T 18.0
void AgentVehicle::setTransform(Transform vehicleTransform, double hdg)
// neue Funktion von Christoph Kirsch um mit Straßenkoordinaten die neue Position des Fahrzeugs zu bestimmten
{
	geometry->setTransform(vehicleTransform, hdg);
}
void AgentVehicle::setPosition(osg::Vec3 &pos, osg::Vec3 &vec)
{
geometry->setTransformByCoordinates(pos, vec);
}
void AgentVehicle::setTransform(osg::Matrix m)
{
	geometry->setTransform(m);
}

void AgentVehicle::makeDecision()
{
    // Lane change decision making //
    //
    if (!dynamic_cast<FiddleRoad *>(currentTransition->road))
    {
        //get static lane set
        std::set<int> staticLaneSet = getStaticLaneSet(s);
        std::vector<ObstacleRelation> vehRel;

        //get neighbouring obstacle relation and dynamic lane set along
        std::map<int, std::vector<ObstacleRelation> > dynamicLaneRelationVectorMap;
        for (std::set<int>::iterator staticLaneSetIt = staticLaneSet.begin(); staticLaneSetIt != staticLaneSet.end(); ++staticLaneSetIt)
        {
            if ((*staticLaneSetIt) == (currentLane - 1) || (*staticLaneSetIt) == (currentLane) || (*staticLaneSetIt) == (currentLane + 1) || ((*staticLaneSetIt) * currentLane < 1 && (abs(*staticLaneSetIt) + abs(currentLane)) <= 2))
            { // Overtake enabling
                vehRel = getNeighboursOnLane(*staticLaneSetIt);
                if (!vehRel[1].vehicle && ((*staticLaneSetIt) == (currentLane) || laneChangeIsSafe(vehRel)))
                {
                    dynamicLaneRelationVectorMap.insert(std::pair<int, std::vector<ObstacleRelation> >(*staticLaneSetIt, vehRel));
                }
            }
        }

        //compute neighbouring vehicle accelerations
        ObstacleRelation thisVehRel(this, 0.0, 0.0);
        std::map<int, double> laneUtilityFractionWithoutThisMap;
        std::map<int, double> laneUtilityFractionWithThisMap;
        for (std::map<int, std::vector<ObstacleRelation> >::iterator mapIt = dynamicLaneRelationVectorMap.begin(); mapIt != dynamicLaneRelationVectorMap.end(); ++mapIt)
        {
            double pFront = vehPars.politeFactor;
            double pRear = vehPars.politeFactor;
            if (mapIt->first == currentLane)
            {
                pRear = 0.0;
            }

            std::vector<ObstacleRelation> relVecWithout = mapIt->second;
            std::vector<double> accVecWithout = computeVehicleAccelerations(relVecWithout);
            laneUtilityFractionWithoutThisMap.insert(std::pair<int, double>(mapIt->first, accVecWithout[0] * pFront + accVecWithout[2] * pRear));

            std::vector<ObstacleRelation> &relVecWith = relVecWithout;
            relVecWith[1] = thisVehRel;
            std::vector<double> accVecWith = computeVehicleAccelerations(relVecWith);
            double newAcc = determineAcceleration(dynamicLaneRelationVectorMap, mapIt->first);
            accVecWith[1] = newAcc;

            laneUtilityFractionWithThisMap.insert(std::pair<int, double>(mapIt->first, accVecWith[0] * pFront + accVecWith[1] + accVecWith[2] * pRear));
        }
        std::map<int, double> vehicleLaneUtilityMap;
        for (std::map<int, std::vector<ObstacleRelation> >::iterator mapIt = dynamicLaneRelationVectorMap.begin(); mapIt != dynamicLaneRelationVectorMap.end(); ++mapIt)
        {
            vehicleLaneUtilityMap[mapIt->first] = laneUtilityFractionWithThisMap[mapIt->first];
            for (std::map<int, std::vector<ObstacleRelation> >::iterator withoutMapIt = dynamicLaneRelationVectorMap.begin(); withoutMapIt != dynamicLaneRelationVectorMap.end(); ++withoutMapIt)
            {
                if (mapIt->first != withoutMapIt->first)
                {
                    vehicleLaneUtilityMap[mapIt->first] += laneUtilityFractionWithoutThisMap[withoutMapIt->first];
                }

                //--- neu Falls Unendlich oder NaN rauskommt, soll es mit 0.0 ersetzt werden
                if (vehicleLaneUtilityMap[mapIt->first] != vehicleLaneUtilityMap[mapIt->first] || vehicleLaneUtilityMap[mapIt->first] > DBL_MAX || vehicleLaneUtilityMap[mapIt->first] < -DBL_MAX)
                {
                    vehicleLaneUtilityMap[mapIt->first] = 0.0;
                }
                //--- neu 20.06.2012
            }
            /*if(name=="schmotzcar") {
            //std::cout << "\t Utility of driving on lane " << mapIt->first << " at this time: " << vehicleLaneUtilityMap[mapIt->first] << std::endl;
         }*/
        }

        //compute end of lane accelerations
        std::map<int, double> eolLaneUtilityMap; //(End Of Lane) lane utility map
        int laneOfMostExtent = currentLane; //lane utiltiy heuristic by determining lane of most extent: Rather a hack than a solution
        double laneOfMostExtentLaneEnd = 0.0;
        //double currentLaneEnd = 0.0;
        bool timeToPanic = false;
        for (std::set<int>::iterator staticLaneSetIt = staticLaneSet.begin(); staticLaneSetIt != staticLaneSet.end(); ++staticLaneSetIt)
        {
            double laneEnd = locateLaneEnd(*staticLaneSetIt,false);
            if (laneOfMostExtentLaneEnd <= (laneEnd - 1.0) || (laneOfMostExtentLaneEnd > (laneEnd - 1.0) && laneOfMostExtentLaneEnd <= (laneEnd + 1.0) && abs(laneOfMostExtent) < abs(*staticLaneSetIt)))
            { //Needs to be reviewed: lane extent tolerance
                laneOfMostExtent = *staticLaneSetIt;
                laneOfMostExtentLaneEnd = laneEnd;
            }
            //alter Code von Florian Seybold - war bereits auskommentiert
            /*if(currentTransition->direction == 1 && currentTransition->road->getId()=="1") {
            std::cout << "Vehicle " << name << ": lane: " << *staticLaneSetIt << ", end: "  << laneEnd << ", laneOME: " << laneOfMostExtent << ", endOME: " << laneOfMostExtentLaneEnd << std::endl;
            std::cout << "laneOfMostExtentLaneEnd <= (laneEnd-1.0): " << (laneOfMostExtentLaneEnd <= (laneEnd-1.0)) << std::endl;
            std::cout << "laneOfMostExtentLaneEnd>(laneEnd-1.0): " << (laneOfMostExtentLaneEnd>(laneEnd-1.0)) << std::endl;
            std::cout << "laneOfMostExtentLaneEnd<=(laneEnd+1.0): " << (laneOfMostExtentLaneEnd<=(laneEnd+1.0)) << std::endl;
            std::cout << "abs(laneOfMostExtent)<abs(laneEnd): " << (abs(laneOfMostExtent)<abs(laneEnd)) << std::endl;
            std::cout << "abs(laneOfMostExtent): " << abs(laneOfMostExtent) << ", abs(laneEnd): " << abs(*staticLaneSetIt) << std::endl;
         }*/

            double accInt = 0.0;
            if (laneEnd < 1e10)
            {
                accInt = vehPars.accMax * (pow(etd(this->du, this->du) / laneEnd, 2.0));
            }
            double accEol = vehPars.accMax * (1 - pow((this->du / velt), vehPars.approachFactor)) - accInt;

            //--- neu Falls Unendlich oder NaN rauskommt, soll es mit 0.0 ersetzt werden
            if (accEol != accEol || accEol > DBL_MAX || accEol < -DBL_MAX)
            {
                accEol = 0.0;
            }
            //--- neu 20.06.2012

            eolLaneUtilityMap.insert(std::pair<int, double>(*staticLaneSetIt, accEol));

            //oncoming end of lane
            if (currentTransition->direction * (*staticLaneSetIt) > 0 && laneEnd < 100.0)
            {
                eolLaneUtilityMap[(*staticLaneSetIt)] -= (100.0 - laneEnd);
            }
        }

        //alter Code von Florian Seybold - war bereits auskommentiert
        //Panic distance
        /*if(laneOfMostExtent != currentLane && currentLaneEnd < vehPars.panicDistance) {
         timeToPanic = true;
      }*/

        /*//stop at signals
      double stopSignalDist = 1e10;
      //current transition
      for(unsigned int i = 0; i<currentTransition->road->getNumRoadSignals(); ++i) {
         RoadSignal* signal = currentTransition->road->getRoadSignal(i);

         //not traffic lights
         if(!(signal->getType()==1000001)) continue;

         //not valid in current direction
         if((signal->getOrientation() + currentTransition->direction)==0) continue;

         //not signaling stop
         if(signal->getValue() > 0.0) continue;

         double thisStopSignalDist = fabs(signal->getS()-u);
         stopSignalDist = std::min(stopSignalDist, thisStopSignalDist);
      }
      //next transition
      RoadTransitionList::iterator nextTransition = currentTransition; ++nextTransition;
      if(nextTransition!=roadTransitionList.end()) {
         for(unsigned int i = 0; i<nextTransition->road->getNumRoadSignals(); ++i) {
            RoadSignal* signal = nextTransition->road->getRoadSignal(i);

            //not traffic lights
            if(!(signal->getType()==1000001)) continue;

            //not valid in current direction
            if((signal->getOrientation() + nextTransition->direction)==0) continue;

            //not signaling stop
            if(signal->getValue() > 0.0) continue;

            double thisStopSignalDist = 0.5*(1+currentTransition->direction)*currentTransition->road->getLength()-currentTransition->direction*u
                                      + 0.5*(1-nextTransition->direction)*nextTransition->road->getLength()+nextTransition->direction*signal->getS();
            stopSignalDist = std::min(stopSignalDist, thisStopSignalDist);
         }
      }
      double accIntStopSignal = 0.0;
      if(stopSignalDist < 1e10) {
         accIntStopSignal = vehPars.accMax*(pow(etd(this->du, this->du) / stopSignalDist, 2.0));
      }
      double accStopSignal = vehPars.accMax*(1 - pow((this->du / velt), vehPars.approachFactor)) - accIntStopSignal;
      if(accStopSignal < acc) {
         acc = accStopSignal;
      }*/

        /*{
         RoadTransitionList::iterator nextTransition = currentTransition; ++nextTransition;
         if(nextTransition!=roadTransitionList.end())
            std::cout << "Vehicle " << name << ": lane of most extent: " << laneOfMostExtent << ", next road: " << nextTransition->road->getId() << std::endl;
      }*/

        std::map<int, double> laneWeightMap; // Enhält Gewichtung der einzelnen Fahrbahnen
        for (std::map<int, std::vector<ObstacleRelation> >::iterator mapIt = dynamicLaneRelationVectorMap.begin(); mapIt != dynamicLaneRelationVectorMap.end(); ++mapIt)
        {
            double laneWeight = 0.0;

            // Neu Andreas 27-11-2012: Falschfahrer auf eigener Spur nicht Überholen, Hack
            //ObstacleRelation obsRel = locateVehicle(currentLane, 1);
            //    if(obsRel.diffDu < 0) {
            // //std::cout<<"Falschfahrer auf eigener Spur"<<endl;
            //	 if(mapIt->first == currentLane) {
            //	 laneWeight +=abs(mapIt->first)*vehPars.lcTreshold*100000.0;
            //	 }
            //}

            if (mapIt->first * currentTransition->direction < 0)
            {
                laneWeight += (abs(mapIt->first) - abs(currentLane) + 1) * 1.25 * vehPars.lcTreshold; //right lane commandment
            }
            else
            {
                laneWeight = -abs(mapIt->first) * vehPars.lcTreshold * 10.0; //avoid oncoming lane
            }

            // Neu Andreas 27-11-2012: geändert von 5*treshold auf 0.5
            if (laneOfMostExtent != Lane::NOLANE && currentTransition->road->getRoadType(this->u) == Road::MOTORWAY)
            { //lane utiltiy heuristic by determining lane of most extent: Rather a hack for motorways exits than a solution
                laneWeight -= pow((double)abs(mapIt->first - laneOfMostExtent), 1) * 0.5 * vehPars.lcTreshold;
            }

            //--> NEU Autobahneinfahrten: Fahrzeuge die gerade an einer Autobahneinfahrt vorbeifahren, sollen nicht auf diese wechseln
            LaneSection *section = currentTransition->road->getLaneSection(this->getU());
            Lane *lane = section->getLane(mapIt->first);
            Lane::LaneType typeOfLane = lane->getLaneType();
            if (typeOfLane == Lane::MWYENTRY)
            {
                laneWeight -= abs(mapIt->first) * 5.0;
            }
            //NEU <-- 19.07.2012

            laneWeightMap.insert(std::pair<int, double>(mapIt->first, laneWeight));
        }

        std::map<int, double> laneUtilityMap;
        for (std::map<int, std::vector<ObstacleRelation> >::iterator mapIt = dynamicLaneRelationVectorMap.begin(); mapIt != dynamicLaneRelationVectorMap.end(); ++mapIt)
        {
            double laneUtil = std::min(vehicleLaneUtilityMap.find(mapIt->first)->second, eolLaneUtilityMap.find(mapIt->first)->second);
            laneUtilityMap.insert(std::pair<int, double>(mapIt->first, laneUtil + laneWeightMap.find(mapIt->first)->second));
        }

        /**--> NEU durchgehende Linie darf auf einer Autobahn von beiden Seiten nicht überfahren werden, jedoch auf der Landstraße darf man nach dem Überholen wieder zurückkommen
       Fahrbahnen mit durchgehender Linie (RoadMark==Solid) bekommen sehr schlechten Gütewert --> wird in der laneUtilityMap überschrieben */
        LaneSection *section = currentTransition->road->getLaneSection(this->getU());
        RoadMark *rmLeft = NULL;
        Lane *LeftLane = section->getLane(currentLane + currentTransition->direction);
        if (LeftLane != NULL)
            rmLeft = LeftLane->getRoadMark(0);
        RoadMark *rmRight = section->getLane(currentLane)->getRoadMark(0); //Roadmark on the right-hand side of the agent

        //gilt für alle Straßen (Autobahn, Landstraße,...)
        for (std::map<int, std::vector<ObstacleRelation> >::iterator mapIt = dynamicLaneRelationVectorMap.begin(); mapIt != dynamicLaneRelationVectorMap.end(); ++mapIt)
        {
            if ((mapIt->first == (currentLane + currentTransition->direction) && rmLeft != NULL && rmLeft->getType() == RoadMark::TYPE_SOLID) || (currentTransition->direction > 0 && currentLane < 0 && rmLeft != NULL && rmLeft->getType() == RoadMark::TYPE_SOLID && mapIt->first > 0)
                || (mapIt->first == (currentLane - currentTransition->direction) && rmRight->getType() == RoadMark::TYPE_SOLID) || (currentTransition->direction < 0 && currentLane > 0 && rmLeft != NULL && rmLeft->getType() == RoadMark::TYPE_SOLID && mapIt->first < 0))
            {
                laneUtilityMap.erase(mapIt->first);
                laneUtilityMap.insert(std::pair<int, double>(mapIt->first, -100000.0)); // durchgezogene Linie nicht überfahren! do not cross solid line!
            }
        }
        // NEU <-- Juli 2012

        if (laneUtilityMap.size() != 0)
        { //Even current lane could be missing in laneUtilityMap, if parking alien vehicles are turning and overlapping into current lane
            std::map<int, double> tmpMap = laneUtilityMap;
            int maximumUtilityLane = laneUtilityMap.begin()->first;
            //std::cout << "Vehicle " << name << ": lane/utility:";
            for (std::map<int, double>::iterator mapIt = laneUtilityMap.begin(); mapIt != laneUtilityMap.end(); ++mapIt)
            {
                //std::cout << " " << mapIt->first << "/" << mapIt->second;
                if (laneUtilityMap[maximumUtilityLane] < mapIt->second)
                {
                    maximumUtilityLane = mapIt->first;
                }
            }

            //alter Code von Florian Seybold - war bereits auskommentiert
            //std::cout << std::endl;
            /*if(name=="schmotzcar") {
           std::cout << "\t Maximum utility of driving on lane " << maximumUtilityLane << std::endl;
           }*/

            if (laneUtilityMap[maximumUtilityLane] > -100000.0 && currentLane != maximumUtilityLane)
            { //--> NEU <--
                currentLane = maximumUtilityLane;
                lcTime = timer; //Zeitpunkt der Entscheidung die Spur zu wechseln
                lcV = this->v; //die Position v auf der Straße, bevor man den Spurwechsel einleitet
            }
        }

        //if(timeToPanic && ((1+currentTransition->direction)/2*currentTransition->road->getLength()-currentTransition->direction*this->u)<10.0) {}
        if (timeToPanic)
        {
            std::cout << "Vehicle " << name << ": panicing --- can't drive on! Road: " << currentTransition->road->getId() << std::endl;
            panicCantReachNextRoad();
        }
    }
}

Road *AgentVehicle::getRoad() const
{
    return currentTransition->road;
}

double AgentVehicle::getU() const
{
    return u;
}

// hinzugefügt Stefan: 24.07.2012
double AgentVehicle::getYaw() const
{
    return hdg;
}

double AgentVehicle::getDu() const
{
    return currentTransition->direction * du;
}

RoadTransition AgentVehicle::getRoadTransition() const
{
    return (*currentTransition);
}

VehicleGeometry *AgentVehicle::getVehicleGeometry()
{
    return geometry;
}

Transform AgentVehicle::getVehicleTransform()
{
    return vehicleTransform;
}

CarGeometry *AgentVehicle::getCarGeometry()
{
    return geometry;
}

const VehicleParameters &AgentVehicle::getVehicleParameters()
{
    return vehPars;
}

void AgentVehicle::setVehicleParameters(const VehicleParameters &vp)
{
    vehPars = vp;
    geometry->setLODrange(vehPars.rangeLOD);
}

double AgentVehicle::getBoundingCircleRadius()
{
    return (geometry == NULL) ? 0.0 : geometry->getBoundingCircleRadius();
}

int AgentVehicle::getLane() const
{
    return currentLane;
}

bool AgentVehicle::isOnLane(int lane) const
{
    double extent = fabs(boundRadius * sin(hdg));
    int lowLane = currentSection->searchLane(u, v - extent);
    if (lowLane == Lane::NOLANE)
    {
        lowLane = currentSection->getTopRightLane();
    }
    int highLane = currentSection->searchLane(u, v + extent);
    if (highLane == Lane::NOLANE)
    {
        highLane = currentSection->getTopLeftLane();
    }

    if (lane >= lowLane && lane <= highLane)
    {
        return true;
    }

    return false;
}

void AgentVehicle::addRouteTransition(const RoadTransition &trans)
{
    routeTransitionList.push_back(trans);
    routeTransitionListBackup.push_back(trans);
}

void AgentVehicle::setRepeatRoute(bool rr)
{
    repeatRoute = rr;
}

/** liefert true, wenn der Spurwechsel auf die Lane, wo sich die Hinternisse vehRelVec befinden, sicher ist; ansonsten false */
bool AgentVehicle::laneChangeIsSafe(std::vector<ObstacleRelation> vehRelVec)
{
    double frontDec = 0;
    if (!vehRelVec[0].noOR())
    {
        double frontVehDu = this->du + vehRelVec[0].diffDu;
        if (frontVehDu * this->du < 0)
        {
            frontDec = vehPars.accMax * (pow(etd(fabs(frontVehDu), -vehRelVec[0].diffDu) / vehRelVec[0].diffU, 2));
        }
    }

    double backDec = 0;
    if (!vehRelVec[2].noOR())
    {
        double backVehDu = this->du + vehRelVec[2].diffDu;
        if (backVehDu * this->du > 0)
        {
            backDec = vehPars.accMax * (pow(etd(fabs(backVehDu), vehRelVec[2].diffDu) / -vehRelVec[2].diffU, 2)); //Verzögerung des hinteren Fahrzeuges
        }
    }

    if ((frontDec > vehPars.decSave) || (backDec > vehPars.decSave))
    { // wenn das vordere bzw. hintere Fahrzeug stärker bremsen muss als die max. zulässige Verzögerung es erlaubt, ist der Spurwechsel nicht erlaubt
        return false;
    }
    else
    {
        if (vehRelVec[0].noOR())
            return true; // No vehicle in front, so safe to change lanes
        else if (vehRelVec[0].vehicle->canPass() && (velt /* our desired velocity */ - vehRelVec[0].vehicle->getU()) > vehPars.passThreshold)
        {
            canBePassed = true;
            return true; // Only pass if the vehicle in front allows passing and distance is large enough
        }
        else
        {
            canBePassed = false; // Not allowed to pass vehicle in front, so other vehicle are not allowed to pass this one either
            // Passing is disabled when vehicles are stopped at a crosswalk and when distance between them is too small, or in general when attempting to pass a vehicle that doesn't allow passing
            return false;
        }
    }

    /*
   if((frontDec>vehPars.decSave) || (backDec>vehPars.decSave)) {
      return false;
   }
   else {
      if (vehRelVec[0].noOR()) return true; // No vehicle in front, so safe to change lanes
      else if (vehRelVec[0].vehicle->canPass() && fabs(vehRelVec[0].vehicle->getU() - this->getU()) > vehPars.passThreshold) {
         canBePassed = true;
         return true; // Only pass if the vehicle in front allows passing and distance is large enough
      }
      else {
         canBePassed = false; // Not allowed to pass vehicle in front, so other vehicle are not allowed to pass this one either
         // Passing is disabled when vehicles are stopped at a crosswalk and when distance between them is too small, or in general when attempting to pass a vehicle that doesn't allow passing
         return false;
      }
   }
   */

/*
   if((frontDec>vehPars.decSave) || (backDec>vehPars.decSave)) {
      return false;
   }
   else {
      if (vehRelVec[0].noOR()) return true;
      else if (vehRelVec[0].vehicle->canPass()) {
         canBePassed = true;
         return true; // Only pass if the vehicle in front allows passing
      }
      else {
         canBePassed = false; // Not allowed to pass vehicle in front, so other vehicle are not allowed to pass this one either
         // Passing is disabled when vehicles are stopped at a crosswalk and when distance between them is too small, or in general when attempting to pass a vehicle that doesn't allow passing
         return false;
      }
   }
*/
}

/** gibt Beschleunigungen einzelner Fahrzeuge zurück, die im Vektor vehRelVec enthalten sind */
std::vector<double> AgentVehicle::computeVehicleAccelerations(std::vector<ObstacleRelation> vehRelVec)
{
    std::vector<double> accVec(vehRelVec.size());

    for (int vehIt = 0; vehIt < vehRelVec.size(); ++vehIt)
    {
        if (vehRelVec[vehIt].noOR())
        {
            accVec[vehIt] = 0;
            continue;
        }
        //int vehPos = (1-vehIt);
        double vehDu = this->du + vehRelVec[vehIt].diffDu;

        int refVehSearchDir = vehDu < 0 ? 1 : -1;
        int refVehIt = -1;
        for (int tmpVehIt = vehIt + refVehSearchDir; (tmpVehIt >= 0) && (tmpVehIt < vehRelVec.size()); tmpVehIt += refVehSearchDir)
        {
            if (!vehRelVec[tmpVehIt].noOR())
            {
                refVehIt = tmpVehIt;
                break;
            }
        }

        if (refVehIt < 0)
        {
            //if(vehIt == 1) {
            accVec[vehIt] = vehPars.accMax * (1 - pow((fabs(vehDu) / velt), vehPars.approachFactor)); //Beschleunigung auf freier Strecke
            //}
            //else {
            //   accVec[vehIt] = 0;
            //}
        }
        else
        {
            double vehDiffDu = -refVehSearchDir * (vehRelVec[vehIt].diffDu - vehRelVec[refVehIt].diffDu);
            double vehDiffU = fabs(vehRelVec[vehIt].diffU - vehRelVec[refVehIt].diffU);
            //Beschleunigung nach differentieller Bewegungsgleichung des Intelligent-Driver Modell
            accVec[vehIt] = vehPars.accMax * (1 - pow((fabs(vehDu) / velt), vehPars.approachFactor) - pow(etd(fabs(vehDu), vehDiffDu) / vehDiffU, 2));
            //accVec[vehIt] = a*(-pow(etd(fabs(vehDu), vehDiffDu) / vehDiffU , 2) );
            //std::cout << "Vehicle " << name << ": vehIt: " << vehIt << ", refVehIt: " << refVehIt << ", vehDu: " << vehDu << ", vehDiffDu: " << vehDiffDu << ", vehDiffU: " << vehDiffU << ", acc: " << accVec[vehIt] << std::endl;
        }
    }

    return accVec;
}

double AgentVehicle::determineAcceleration(const std::map<int, std::vector<ObstacleRelation> > &relMap, int lane)
{
    std::set<double> accSet;

    ObstacleRelation obsRel = relMap.find(lane)->second.at(0);
    if (!obsRel.noOR())
    {
        double vehDiffDu = -obsRel.diffDu;
        double vehDiffU = fabs(obsRel.diffU);
        //std::cout << "Vehicle " << name << ": vehicle at front on lane " << lane << ": " << obsRel.vehicle->getName() << " - " << vehDiffU << " - " << vehDiffDu << ", ";
        accSet.insert(vehPars.accMax * (1 - pow((this->du / velt), vehPars.approachFactor) - pow(etd(this->du, vehDiffDu) / vehDiffU, 2)));
    }
    else
    {
        //std::cout << "Vehicle " << name << ": no vehicle at front on lane " << lane << ", ";
        accSet.insert(vehPars.accMax * (1 - pow((fabs(this->du) / velt), vehPars.approachFactor)));
    }

    //std::cout << "Vehicle " << name << ": Accs: Lane " << lane << ": " << *accSet.begin();
    for (std::map<int, std::vector<ObstacleRelation> >::const_iterator relMapIt = relMap.begin(); relMapIt != relMap.end(); ++relMapIt)
    {
        if (abs(relMapIt->first) >= abs(lane) || relMapIt->first * lane < 0)
        {
            continue;
        }

        ObstacleRelation obsRel = relMapIt->second.at(1);
        if (obsRel.noOR())
        {
            obsRel = relMapIt->second.at(0);
        }
        if (!obsRel.noOR())
        {
            double refVehDu = this->du + obsRel.diffDu;
            accSet.insert(vehPars.accMax * (1 - pow((this->du / refVehDu), vehPars.approachFactor)));
        }
    }
    //std::cout << std::endl;

    if (accSet.size() != 0)
    {
        return (*std::min_element(accSet.begin(), accSet.end()));
    }
    else
    {
        return 0.0;
    }
}

/** gibt den vorderen Nachbarn gespreichert in vehRelVec[0] und den hinterern Nachbarn in vehRelVec[2] auf der Spur=lane zurück */
std::vector<ObstacleRelation> AgentVehicle::getNeighboursOnLane(int lane)
{
    std::vector<ObstacleRelation> vehRelVec(3);

    ObstacleRelation vehRelFront = locateVehicle(lane, 1); //lokalisiere Fahrzeug auf der Spur "lane" vor dem Agent

    if (vehRelFront.vehicle && vehRelFront.vehicle != this)
    {
        if (vehRelFront.diffU == 0.0)
        {
            vehRelVec[1] = vehRelFront;
            //return vehRelVec;
        }
        else
        {
            //double laneEndDiffU = locateLaneEnd(lane);
            //if(vehRelFront.diffU < laneEndDiffU) {
            vehRelVec[0] = vehRelFront;
            //}
            //else {
            //   vehRelVec[0] = ObstacleRelation(ObstacleRelation::ENDOFLANE, laneEndDiffU, -this->du);
            //}
        }
    }
    /*else {
      double laneEndDiffU = locateLaneEnd(lane);
      if(laneEndDiffU < 1e10) {
         vehRelVec[0] = ObstacleRelation(ObstacleRelation::ENDOFLANE, laneEndDiffU, -this->du);
      }
   }*/

    ObstacleRelation vehRelRear = locateVehicle(lane, -1); //lokalisiere Fahrzeug auf der Spur "lane" hinter dem Agent
    if (vehRelRear.vehicle && vehRelRear.vehicle != this)
    {
        if (vehRelRear.diffU == 0.0)
        {
            vehRelVec[1] = vehRelRear;
            return vehRelVec;
        }
        else
        {
            vehRelVec[2] = vehRelRear;
        }
    }

    /*std::cout << "Vehicle " << name << ": neighbours on lane " << lane << ":";
   if(vehRelVec[0].vehicle)
      std::cout << " front: " << vehRelVec[0].vehicle->getName() << " " << vehRelVec[0].diffU << " " << vehRelVec[0].diffDu << ",";
   if(vehRelVec[1].vehicle)
      std::cout << " beside: " << vehRelVec[1].vehicle->getName() << vehRelVec[1].diffU << " " << vehRelVec[1].diffDu << ",";
   if(vehRelVec[2].vehicle)
      std::cout << " rear: " << vehRelVec[2].vehicle->getName() << vehRelVec[2].diffU << " " << vehRelVec[2].diffDu << ",";
   std::cout << std::endl;*/

    return vehRelVec;
}

std::set<int> AgentVehicle::getStaticLaneSet(double s)
{
    RoadTransitionPoint point = getRoadTransitionPoint(s);
    LaneSection *section = point.transition->road->getLaneSection(point.u);

    int fromLane = 0;
    int toLane = 0;
    int laneInc = 0;
    if (point.transition->direction < 0)
    {
        fromLane = section->getTopLeftLane();
        toLane = section->getTopRightLane();
        laneInc = -1;
    }
    else
    {
        fromLane = section->getTopRightLane();
        toLane = section->getTopLeftLane();
        laneInc = 1;
    }

    bool lastLaneDrivable = true;
    std::set<int> laneSet;
    //std::cout << "Vehicle " << name << ": static lane set:";
    for (int laneIt = fromLane; laneIt != toLane + laneInc; laneIt += laneInc)
    {
        if (!lastLaneDrivable && laneIt * laneInc >= 0)
        {
            break;
        }
        Lane *lane = section->getLane(laneIt);
        if (laneIt != 0 && lane)
        {
            Lane::LaneType type = lane->getLaneType();
            if (drivableLaneTypeSet.find(type) != drivableLaneTypeSet.end())
            {
                //std::cout << " " << laneIt;
                laneSet.insert(laneIt);
                lastLaneDrivable = true;
            }
            else
            {
                lastLaneDrivable = false;
            }
        }
    }
    //std::cout << std::endl;

    return laneSet;
}

/** gibt die Straße und entsprechenden u-Wert zurück */
RoadTransitionPoint AgentVehicle::getRoadTransitionPoint(double sPoint)
{
    double sDiff = sPoint - s;
    double sLook = 0;
    int searchDir = 1;
    if (sDiff < 0)
    {
        searchDir = -1;
    }
    else if (sDiff == 0)
    {
        return (RoadTransitionPoint(currentTransition, u));
    }

    RoadTransitionList::iterator transIt = currentTransition;

    sLook += (1 + searchDir * transIt->direction) / 2 * transIt->road->getLength() - searchDir * transIt->direction * u;

    while (sLook < sDiff)
    {
        if (searchDir < 0 && (transIt-- == roadTransitionList.begin()))
        {
            return RoadTransitionPoint(roadTransitionList.end(), 0.0);
        }
        else if (searchDir > 0 && (++transIt == roadTransitionList.end()))
        {
            return RoadTransitionPoint(roadTransitionList.end(), 0.0);
        }
        sLook += transIt->road->getLength();
    }

    double u = (1 + searchDir * transIt->direction) / 2 * transIt->road->getLength() - searchDir * transIt->direction * (sLook - sDiff);
    return (RoadTransitionPoint(transIt, u));
}

void AgentVehicle::panicCantReachNextRoad()
{
    RoadTransitionList::iterator nextTransition = currentTransition;
    ++nextTransition;

    if (nextTransition == roadTransitionList.end() || nextTransition->junction == NULL)
    {
        //vanish();
        return;
    }
    else
    {
        Junction *junction = nextTransition->junction;
        Road *unreachableRoad = nextTransition->road;

        PathConnectionSet connSet = junction->getPathConnectionSet(currentTransition->road);
        for (PathConnectionSet::iterator connIt = connSet.begin(); connIt != connSet.end(); ++connIt)
        {
            if ((*connIt)->getConnectingPath() == unreachableRoad)
            {
                connSet.erase(connIt);
                break;
            }
        }
        if (connSet.empty())
        {
            //vanish();
            return;
        }

        roadTransitionList.erase(nextTransition, roadTransitionList.end());
        DetermineNextRoadVehicleAction::removeAllActions(this);

        //int path = rand() % connSet.size();
        int path = TrafficSimulation::instance()->getIntegerRandomNumber() % connSet.size();
        PathConnectionSet::iterator connSetIt = connSet.begin();
        std::advance(connSetIt, path);
        PathConnection *conn = *connSetIt;
        RoadTransition newTrans(conn->getConnectingPath(), conn->getConnectingPathDirection(), junction);
        roadTransitionList.push_back(newTrans);
        vehiclePositionActionMap.insert(std::pair<double, VehicleAction *>(s, new DetermineNextRoadVehicleAction()));
    }
}

void AgentVehicle::vanish()
{
    std::cout << "Vehicle " << name << " vanishing! Goodbye..." << std::endl;
    TrafficSimulation::instance()->getVehicleManager()->removeVehicle(this, currentTransition->road);
}

bool AgentVehicle::planRoute()
{
    bool oneFound = false;
    while (!routeTransitionList.empty())
    {
        bool found = findRoute(routeTransitionList.front());
        oneFound = oneFound || found;
        routeTransitionList.pop_front();
    }

    return oneFound;
}

bool AgentVehicle::findRoute(RoadTransition goalTrans)
{
    double goalS = (goalTrans.direction < 0) ? goalTrans.road->getLength() : 0.0;
    Vector3D goalPos3D = goalTrans.road->getChordLinePlanViewPoint(goalS);
    Vector2D goalPos(goalPos3D.x(), goalPos3D.y());

    std::set<RouteFindingNode *, RouteFindingNodeCompare> fringeNodeSet;
    RouteFindingNode *baseNode = new RouteFindingNode(roadTransitionList.back(), goalTrans, goalPos);
    if (baseNode->foundGoal())
    {
        roadTransitionList.splice(roadTransitionList.end(), baseNode->backchainTransitionList());
        delete baseNode;
        return true;
    }
    else if (baseNode->isEndNode())
    {
        delete baseNode;
        return false;
    }
    fringeNodeSet.insert(baseNode);

    while (!fringeNodeSet.empty())
    {
        std::set<RouteFindingNode *, RouteFindingNodeCompare>::iterator currentNodeIt = fringeNodeSet.begin();
        RouteFindingNode *currentNode = (*currentNodeIt);
        Junction *currentJunction = currentNode->getEndJunction();
        PathConnectionSet connSet = currentJunction->getPathConnectionSet(currentNode->getLastTransition().road);
        for (PathConnectionSet::iterator connSetIt = connSet.begin(); connSetIt != connSet.end(); ++connSetIt)
        {

            RouteFindingNode *expandNode = new RouteFindingNode(RoadTransition((*connSetIt)->getConnectingPath(), (*connSetIt)->getConnectingPathDirection(), currentJunction), goalTrans, goalPos, currentNode);
            currentNode->addChild(expandNode);

            if (expandNode->foundGoal())
            {
                roadTransitionList.splice(roadTransitionList.end(), expandNode->backchainTransitionList());
                delete baseNode;
                return true;
            }

            else if (!(expandNode->isEndNode()))
            {
                fringeNodeSet.insert(expandNode);
            }
        }

        fringeNodeSet.erase(currentNodeIt);
    }

    std::cerr << "Vehicle " << name << ": findRoute(): Goal state not found using complete A*-search! So no route available to road " << goalTrans.road->getId() << " with direction " << goalTrans.direction << ", sorry..." << std::endl;
    return false;
}

/** gibt ein Hindernis-Fahrzeug mit seinem Relativabstand und seiner Relativgeschwindigkeit zum Agent zurück
    auf der Spur=lane und vehOffset=-1 für Hindernis hinter dem Agent; vehOffset=1 vor dem Agent und vehOffset=0 neben dem Agent */
ObstacleRelation AgentVehicle::locateVehicle(int lane, int vehOffset)
{
    if (vehOffset == 0)
    {
        if (lane == currentLane)
        {
            return ObstacleRelation(this, 0.0, 0.0);
        }
        else
        {
            return ObstacleRelation(NULL, 0.0, 0.0);
        }
    }

    double dis = 0;
    double dvel = 0;
    int dirSearch = (vehOffset < 0) ? -1 : 1;
    vehOffset = abs(vehOffset);
    int numVeh = 0;

    //RoadTransitionList::iterator transIt = roadTransitionList.begin();
    RoadTransitionList::iterator transIt = currentTransition;
    int dirTrans = transIt->direction * dirSearch;
    const VehicleList &vehList = VehicleManager::Instance()->getVehicleList(transIt->road);
    bool vehItDirForward = true;
    if (transIt->direction * dirSearch < 0)
    {
        //vehList.reverse();
        vehItDirForward = false;
    }
    VehicleList::const_iterator vehIt = std::find(vehList.begin(), vehList.end(), this);
    VehicleList::const_iterator nextVehIt = vehIt;
    double nextVehOldU = (*nextVehIt)->getU();
    if (vehItDirForward)
    {
        ++nextVehIt;
    }
    else
    {
        if (nextVehIt != vehList.begin())
        {
            --nextVehIt;
        }
        else
        {
            nextVehIt = vehList.end();
        }
    }
    while (nextVehIt != vehList.end() && lane != Lane::NOLANE)
    {
        double nextVehNewU = (*nextVehIt)->getU();
        lane = transIt->road->traceLane(lane, nextVehOldU, nextVehNewU);
        if (lane == Lane::NOLANE)
        {
            return ObstacleRelation(NULL, 0, 0);
        }
        nextVehOldU = nextVehNewU;
        //if( ((*nextVehIt)->getLane() == lane) && (++numVeh==vehOffset)) {
        if (((*nextVehIt)->isOnLane(lane)) && (++numVeh == vehOffset))
        {
            break;
        }
        if (vehItDirForward)
            ++nextVehIt;
        else if (nextVehIt == vehList.begin())
        {
            nextVehIt = vehList.end();
            break;
        }
        else
            --nextVehIt;
    }

    if (nextVehIt == vehList.end())
    {
        dis += (1 - dirTrans) / 2 * transIt->road->getLength();

        while (true)
        {
            dis += dirTrans * transIt->road->getLength();
            if (lane != Lane::NOLANE)
            {
                int newLane = transIt->road->traceLane(lane, nextVehOldU, (transIt->direction * dirSearch < 0) ? 0.0 : transIt->road->getLength());
                if (newLane != Lane::NOLANE)
                {
                    lane = newLane;
                }
                //if(lane==Lane::NOLANE) {
                //   return ObstacleRelation(NULL,0,0);
                //}
            }
            RoadTransitionList::iterator lastTransIt = transIt;
            if ((dirSearch < 0) ? ((transIt != roadTransitionList.begin() && (--transIt) == transIt)) : ((++transIt) != roadTransitionList.end()))
            {
                //Junction* junction = transIt->road->getJunction();
                Junction *junction = transIt->junction;
                if (junction)
                {
                    lane = junction->getConnectingLane(lastTransIt->road, transIt->road, lane);
                    if (lane == Lane::NOLANE)
                    {
                        return ObstacleRelation(NULL, 0, 0);
                    }
                }
                const VehicleList &vehList = VehicleManager::Instance()->getVehicleList((transIt)->road);
                bool vehItDirForward = true;
                if (transIt->direction * dirSearch < 0)
                {
                    //vehList.reverse();
                    vehItDirForward = false;
                }
                nextVehOldU = 0.0;
                if (vehList.begin() != vehList.end())
                {
                    if (vehItDirForward)
                        nextVehIt = vehList.begin();
                    else
                        nextVehIt = --vehList.end();
                    while (nextVehIt != vehList.end() && lane != Lane::NOLANE)
                    {
                        double nextVehNewU = (*nextVehIt)->getU();
                        lane = transIt->road->traceLane(lane, nextVehOldU, nextVehNewU);
                        if (lane == Lane::NOLANE)
                        {
                            return ObstacleRelation(NULL, 0, 0);
                        }
                        nextVehOldU = nextVehNewU;
                        //if( ((*nextVehIt)->getLane() == lane) && (++numVeh==vehOffset) )
                        if (((*nextVehIt)->isOnLane(lane)) && (++numVeh == vehOffset))
                        {
                            //std::cout << "Lane " << lane << ", dirSearch: " << dirSearch << ": found vehicle: " << (*nextVehIt)->getName() << std::endl;
                            break;
                        }
                        if (vehItDirForward)
                            ++nextVehIt;
                        else if (nextVehIt == vehList.begin())
                        {
                            nextVehIt = vehList.end();
                            break;
                        }
                        else
                            --nextVehIt;
                        //++nextVehIt;
                    }
                }
                else
                {
                    continue;
                }

                if (nextVehIt != vehList.end())
                    break;
            }
            else
            {
                //std::cout << "Lane " << lane << ", dirSearch: " << dirSearch << ": road transition list size: " << roadTransitionList.size() << std::endl;
                return ObstacleRelation(NULL, 0, 0);
            }
        }
        dis += (1 - transIt->direction * dirSearch) / 2 * dirTrans * transIt->road->getLength();
        dis += dirTrans * transIt->direction * dirSearch * (*nextVehIt)->getU();
        double bodyExtent = boundRadius + (*nextVehIt)->getBoundingCircleRadius();
        dis = dirSearch * dirTrans * (dis - this->u) - dirSearch * bodyExtent;
        if (dirSearch * dis < 0)
        {
            dis = 0.0;
        }
        dvel = transIt->direction * (*nextVehIt)->getDu() - this->du;
    }
    else
    {
        double bodyExtent = boundRadius + (*nextVehIt)->getBoundingCircleRadius();
        dis = dirSearch * dirTrans * ((*nextVehIt)->getU() - this->u) - dirSearch * bodyExtent;
        if (dirSearch * dis < 0)
        {
            dis = 0.0;
        }
        dvel = dirSearch * dirTrans * (*nextVehIt)->getDu() - this->du;
    }

    return ObstacleRelation((*nextVehIt), dis, dvel);
    //std::cout << "First vehicle of road " << currentTransition->road->getId() << ": " << vehList.front()->getName() << std::endl;
}

double AgentVehicle::locateLaneEnd(int lane,bool resetIfLaneEnds)
{
    //std::cout << "Locate end of lane " << lane << ": successors: ";

    double pos = 0;
    bool entry = true;
    if (roadTransitionList.size() == 0)
    {
        bool isSignal = false;
        return currentTransition->road->getLaneEnd(lane, u, currentTransition->direction, isSignal);
    }

    for (RoadTransitionList::iterator transIt = currentTransition; transIt != roadTransitionList.end(); ++transIt)
    {
        if (dynamic_cast<FiddleRoad *>(transIt->road))
        {
            return 1e10;
        }
        int newLane = lane;
        double startU = 0;
        if (entry)
        {
            startU = u;
            entry = false;
        }
        else
        {
            startU = (1 - transIt->direction) / 2 * transIt->road->getLength();
        }
        bool isSignal = false;
        double roadPos = transIt->road->getLaneEnd(newLane, startU, transIt->direction, isSignal);
        pos += (1.0 - transIt->direction) * 0.5 * transIt->road->getLength() + transIt->direction * roadPos;
        //std::cout << "\t Road: " << transIt->road->getId() << ", lane: " << lane << ", new lane: " << newLane << ", roadPos: " << roadPos << ", pos: " << pos << ", startU: " << startU << std::endl;
        //std::cout << newLane;

        if (isSignal)
        {
            lane = Lane::NOLANE;
        }
        else
        {
            RoadTransitionList::iterator nextTransIt = transIt;
            ++nextTransIt;
            //if((roadPos >= transIt->road->getLength() || roadPos<=0.0) && nextTransIt!=roadTransitionList.end() && nextTransIt->road->isJunctionPath() ) {
            //if(nextTransIt!=roadTransitionList.end() && nextTransIt->road->isJunctionPath() ) {
            if (nextTransIt != roadTransitionList.end() && nextTransIt->junction)
            {
                //Junction* junction = nextTransIt->road->getJunction();
                Junction *junction = nextTransIt->junction;
                lane = junction->getConnectingLane(transIt->road, nextTransIt->road, lane);
                if (lane == Lane::NOLANE)
                {
                    lane = newLane; // we reached a juction and did not find a connectiong lane for the path we planned earlier, thus we will try to plan a new path
                    // we clear the road transition list and create a new one
                    if(resetIfLaneEnds)
                    {
                        RoadTransitionList::iterator it = roadTransitionList.end();
                        it--;
                        RoadTransitionList::iterator toRemove;
                        while(it !=currentTransition)
                        {
                            toRemove = it;
                            it--;
                            roadTransitionList.erase(toRemove); 
                        }
                       // RoadTransition t = *currentTransition;
                        //roadTransitionList.clear();
                        //roadTransitionList.push_back(t);
                        //currentTransition = roadTransitionList.begin();
                        //s=u;
                        vehiclePositionActionMap.insert(std::pair<double, VehicleAction *>(s, new DetermineNextRoadVehicleAction())); // s are the total m so far (not u)
                        sowedDetermineNextRoadVehicleAction = true;
                        executeActionMap();
                        return locateLaneEnd(lane,false);
                    }
                }
            }
            else
            {
                lane = newLane;
            }
        }
        //std::cout << "\t Road: " << transIt->road->getId() << ", lane: " << lane << ", new lane: " << newLane << std::endl;

        if (lane == Lane::NOLANE)
        {
            pos -= (1 - currentTransition->direction) / 2 * currentTransition->road->getLength() + currentTransition->direction * u;
            return pos;
        }
    }

    return 1e10;
}

void AgentVehicle::extendSignalBarrierList(const RoadTransition &rt)
{
    std::multimap<double, std::pair<RoadTransition, RoadSignal *> > roadSignalMap;

    for (unsigned int signalIt = 0; signalIt < rt.road->getNumRoadSignals(); ++signalIt)
    {
        RoadSignal *signal = rt.road->getRoadSignal(signalIt);
        if (signal->getType() == 1000003 && (signal->getSubtype() == 1 || signal->getSubtype() == 2)
            && (signal->getOrientation() == RoadSignal::BOTH_DIRECTIONS
                || (signal->getOrientation() == RoadSignal::POSITIVE_TRACK_DIRECTION && rt.direction == 1)
                || (signal->getOrientation() == RoadSignal::NEGATIVE_TRACK_DIRECTION && rt.direction == -1)))
        {
            roadSignalMap.insert(std::make_pair(signal->getS(), std::make_pair(rt, signal)));
        }
    }

    if (rt.direction == 1)
    {
        for (std::multimap<double, std::pair<RoadTransition, RoadSignal *> >::iterator signalIt = roadSignalMap.begin(); signalIt != roadSignalMap.end(); ++signalIt)
        {
            signalBarrierList.push_back(signalIt->second);
        }
    }
    else
    {
        for (std::multimap<double, std::pair<RoadTransition, RoadSignal *> >::reverse_iterator signalIt = roadSignalMap.rbegin(); signalIt != roadSignalMap.rend(); ++signalIt)
        {
            signalBarrierList.push_back(signalIt->second);
        }
    }
}

double AgentVehicle::getAcceleration(const ObstacleRelation &obsRel, const double &laneEndFix, const double &du, const double &deltaU, const double &deltaDu) const
{
    double acc = 0;

    // Determine obstacles and resulting acceleration on current lane //
    //
    //ObstacleRelation vehRel = getNeighboursOnLane(currentLane);

    if (!obsRel.noOR())
    {
        double vehDiffDu = -obsRel.diffDu - deltaDu;
        double vehDiffU = fabs(obsRel.diffU - deltaU);
        acc = vehPars.accMax * (1 - pow((du / velt), vehPars.approachFactor) - pow(etd(du, vehDiffDu) / vehDiffU, 2));
    }
    else
    {
        acc = vehPars.accMax * (1 - pow((fabs(du) / velt), vehPars.approachFactor));
    }

    double accInt = 0.0;
    if (laneEndFix < 1e10)
    {
        double laneEnd = laneEndFix - deltaU;
        accInt = vehPars.accMax * (pow(etd(du, du) / laneEnd, 2.0));
    }
    double accEol = vehPars.accMax * (1 - pow((du / velt), vehPars.approachFactor)) - accInt;

    if (accEol < acc)
    {
        acc = accEol;
    }

    return acc;
}

/** berechnet den effektiven Wunschabstand */
double AgentVehicle::etd(double vel, double dvel) const
{
    //vel = fabs(vel);
    if (vel < 0.0)
    {
        return vehPars.deltaSmin;
    }

    /*
   etd...		effektiver Wunschabstand
   deltaSmin...	minimaler Abstand
   respTime...	zeitlicher Sicherheitsabstand
   vel...		Geschwindigkeit des Agenten
   dvel...		Geschwindigkeit des Hindernisses relativ zum Fahrzeugagent
   accMax...	maximale Beschleunigung
   decComf...	komfortable Bremsverzögerung (ist nicht die maximale Verzögerung des Fahrzeugagenten,
				vielmehr wird so stark wie nötig verzögert)
   */
    //double etd =  vehPars.deltaSmin + vehPars.respTime*vel + (vel*dvel)/(2*sqrt(vehPars.accMax*vehPars.decComf));
    //return (etd > vehPars.deltaSmin) ? etd :  vehPars.deltaSmin;

    //NEU bei Autobahneinfahrt darf der Mindestabstand um einen Meter verkleinert werden
    LaneSection *section = currentTransition->road->getLaneSection(this->getU());
    Lane *lane = section->getLane(currentLane);
    Lane::LaneType type = lane->getLaneType();
    double minDist = vehPars.deltaSmin;
    if (type == Lane::MWYENTRY)
    {
        minDist = vehPars.deltaSmin - 1.0;
        minDist = (minDist > 0) ? minDist : vehPars.deltaSmin;
    }

    double etd = minDist + vehPars.respTime * vel + (vel * dvel) / (2 * sqrt(vehPars.accMax * vehPars.decComf));
    return (etd > minDist) ? etd : minDist;
}

/** führt Aktionen wegabhängig und zeitabhängig aus */
bool AgentVehicle::executeActionMap()
{
    bool actionPerformed = false;

    VehicleActionMap::iterator actionEnd = vehiclePositionActionMap.upper_bound(s);
    for (VehicleActionMap::iterator actionIt = vehiclePositionActionMap.begin(); actionIt != actionEnd;)
    {
        (*(actionIt->second))(this);
        delete actionIt->second;
        vehiclePositionActionMap.erase(actionIt++);
        actionPerformed = true;
        actionEnd = vehiclePositionActionMap.upper_bound(s);
    }

    actionEnd = vehicleTimerActionMap.upper_bound(timer);
    for (VehicleActionMap::iterator actionIt = vehicleTimerActionMap.begin(); actionIt != actionEnd;)
    {
        (*(actionIt->second))(this);
        delete actionIt->second;
        vehicleTimerActionMap.erase(actionIt++);
        actionPerformed = true;
        actionEnd = vehicleTimerActionMap.upper_bound(timer);
    }

    return actionPerformed;
}

/** sucht sich zufällig die nächste Straße aus und bestimmt Aktionen für die nächste Straße */
void DetermineNextRoadVehicleAction::operator()(AgentVehicle *veh)
{
    //std::cout << "Determining next road!" << std::endl;
    RoadTransition trans = veh->roadTransitionList.back();

    TarmacConnection *conn = NULL;
    if (veh->roadTransitionList.back().direction > 0)
    {
        conn = veh->roadTransitionList.back().road->getSuccessorConnection();
    }
    else if (veh->roadTransitionList.back().direction < 0)
    {
        conn = veh->roadTransitionList.back().road->getPredecessorConnection();
    }
    if (conn)
    {
        Junction *junction;
        Road *road;
        double isJunctionPath = false;
        RoadTransition newTrans = trans;
        int indicator = 0;

        bool motorwayExit = false;
        if ((road = dynamic_cast<Road *>(conn->getConnectingTarmac())))
        {
            newTrans.road = road;
            newTrans.direction = conn->getConnectingTarmacDirection();
            newTrans.junction = NULL;
            /*if(newTrans->road->getId()=="1fr") {
            std::cout << "Vehicle " << name << ":" << std::endl;
            std::cout << "\t\tCurrent road is id " << roadTransitionList.back().road->getId() << ": current lane: " << roadTransitionList.back().lane << ", current roadTransitionList.back().direction: " << roadTransitionList.back().direction << std::endl;
            std::cout << "\t\tChoice road is id " << newTrans->road->getId() << ": lane of choice: " << newTrans->lane << ", roadTransitionList.back().direction of choice: " << newTrans->direction << std::endl;
         }*/
        }
        else if ((junction = dynamic_cast<Junction *>(conn->getConnectingTarmac())))
        {
            PathConnectionSet connSet = junction->getPathConnectionSet(veh->roadTransitionList.back().road,veh->currentLane);
            if(connSet.size()==0)
            {
                connSet = junction->getPathConnectionSet(veh->roadTransitionList.back().road);
            }
            /*int path = rand() % connSet.size();
         PathConnectionSet::iterator connSetIt = connSet.begin();
         std::advance(connSetIt, path);
         PathConnection* conn = *connSetIt;*/
            PathConnection *conn = connSet.choosePathConnection(TrafficSimulation::instance()->getZeroOneRandomNumber());
            if(conn)
            {
                newTrans.road = conn->getConnectingPath();
                newTrans.direction = conn->getConnectingPathDirection();
                newTrans.junction = junction;
                indicator = conn->getConnectingPathIndicator();
                isJunctionPath = true;
                LaneSection *section = newTrans.road->getLaneSection((1 - newTrans.direction) / 2 * newTrans.road->getLength());
                for (int laneIt = section->getTopRightLane(); laneIt <= section->getTopLeftLane(); ++laneIt)
                {
                    if (section->getLane(laneIt)->getLaneType() == Lane::MWYEXIT)
                    {
                        motorwayExit = true;
                        break;
                    }
                }
            }
        }
        else
        {
            newTrans.direction = 0;
        }

        veh->roadTransitionList.push_back(newTrans);

        // NEXT ROAD DETERMINATION ACTION //
        //
        double transLength = veh->roadTransitionList.getLength(veh->currentTransition, veh->roadTransitionList.end())
                             + 0.5 * (veh->currentTransition->direction - 1) * veh->currentTransition->road->getLength()
                             - veh->currentTransition->direction * veh->u;
        veh->vehiclePositionActionMap.insert(std::pair<double, VehicleAction *>(veh->s + transLength - veh->minTransitionListExtent, new DetermineNextRoadVehicleAction()));

        // INDICATOR //
        //
        if (indicator != 0)
        {
            veh->vehiclePositionActionMap.insert(std::pair<double, VehicleAction *>(veh->s + transLength - newTrans.road->getLength() - 50, new JunctionIndicatorVehicleAction(indicator))); // indicator on
            veh->vehiclePositionActionMap.insert(std::pair<double, VehicleAction *>(veh->s + transLength + 10, new JunctionIndicatorVehicleAction(0))); // indicator off
        }

        /*if(isJunctionPath && newTrans.road->getRoadType((1-newTrans.direction)/2*newTrans.road->getLength())!=Road::MOTORWAY) {
         veh->junctionPathLookoutList.push_back(--veh->roadTransitionList.end());
         veh->vehiclePositionActionMap.insert( std::pair<double , VehicleAction*>(veh->s + transLength - veh->roadTransitionList.back().road->getLength() - veh->junctionWaitTriggerDist, new WaitAtJunctionVehicleAction()) );
         //std::cout << "Junction Trigger at s: " << veh->s + transLength - veh->roadTransitionList.back().road->getLength() - veh->junctionWaitTriggerDist << " with veh->s: " << veh->s << ", transLength: " << transLength << ", road length: " <<  veh->roadTransitionList.back().road->getLength() << ", trigger dist: " << veh->junctionWaitTriggerDist << std::endl;
      }*/

        /*if(motorwayExit) {
         veh->vehiclePositionActionMap.insert( std::pair<double , VehicleAction*>(veh->s + transLength - veh->roadTransitionList.back().road->getLength() - 500, new SetDrivingStateVehicleAction(AgentVehicle::EXIT_MOTORWAY)) );
      }*/
    }
}

int DetermineNextRoadVehicleAction::removeAllActions(AgentVehicle *veh)
{
    int erased = 0;

    for (VehicleActionMap::iterator actionIt = veh->vehiclePositionActionMap.begin(); actionIt != veh->vehiclePositionActionMap.end();)
    {
        if (dynamic_cast<DetermineNextRoadVehicleAction *>(actionIt->second))
        {
            veh->vehiclePositionActionMap.erase(actionIt++);
            ++erased;
        }
        else
        {
            ++actionIt;
        }
    }

    for (VehicleActionMap::iterator actionIt = veh->vehicleTimerActionMap.begin(); actionIt != veh->vehicleTimerActionMap.end();)
    {
        if (dynamic_cast<DetermineNextRoadVehicleAction *>(actionIt->second))
        {
            veh->vehicleTimerActionMap.erase(actionIt++);
            ++erased;
        }
        else
        {
            ++actionIt;
        }
    }

    return erased;
}

int JunctionIndicatorVehicleAction::removeAllActions(AgentVehicle *veh)
{
    int erased = 0;

    for (VehicleActionMap::iterator actionIt = veh->vehiclePositionActionMap.begin(); actionIt != veh->vehiclePositionActionMap.end();)
    {
        if (dynamic_cast<JunctionIndicatorVehicleAction *>(actionIt->second))
        {
            veh->vehiclePositionActionMap.erase(actionIt++);
            ++erased;
        }
        else
        {
            ++actionIt;
        }
    }

    for (VehicleActionMap::iterator actionIt = veh->vehicleTimerActionMap.begin(); actionIt != veh->vehicleTimerActionMap.end();)
    {
        if (dynamic_cast<JunctionIndicatorVehicleAction *>(actionIt->second))
        {
            veh->vehicleTimerActionMap.erase(actionIt++);
            ++erased;
        }
        else
        {
            ++actionIt;
        }
    }

    return erased;
}

void JunctionIndicatorVehicleAction::operator()(AgentVehicle *veh)
{
    // called 50m before junction to turn on and 10m after junction to turn off indicators
    // indicator_ is class member of JunctionIndicatorVehicleAction

    VehicleState &vehState = veh->getVehicleState();

    if (indicator_ == -1)
    {
        vehState.junctionState = VehicleState::JUNCTION_RIGHT;
    }
    else if (indicator_ == 1)
    {
        vehState.junctionState = VehicleState::JUNCTION_LEFT;
    }
    else
    {
        vehState.junctionState = VehicleState::JUNCTION_NONE;
    }
}

//NEU 02-02-11
ObstacleData::ObstacleTypeEnum AgentVehicle::getVehType()
{

    if (vehPars.obstacleType == "car")
        return ObstacleData::OBSTACLE_CAR;
    else if (vehPars.obstacleType == "van")
        return ObstacleData::OBSTACLE_VAN;
    else if (vehPars.obstacleType == "truck")
        return ObstacleData::OBSTACLE_TRUCK;
    else if (vehPars.obstacleType == "suv")
        return ObstacleData::OBSTACLE_SUV;
    else if (vehPars.obstacleType == "scar")
        return ObstacleData::OBSTACLE_SPORTSCAR;
    else if (vehPars.obstacleType == "small")
        return ObstacleData::OBSTACLE_SMALLCAR;
    else if (vehPars.obstacleType == "police")
        return ObstacleData::OBSTACLE_POLICE;
    else if (vehPars.obstacleType == "human")
        return ObstacleData::OBSTACLE_HUMAN;
    else if (vehPars.obstacleType == "tractor")
        return ObstacleData::OBSTACLE_TRACTOR;
    else if (vehPars.obstacleType == "bicycle")
        return ObstacleData::OBSTACLE_BICYCLE;
    else if (vehPars.obstacleType == "motorcycle")
        return ObstacleData::OBSTACLE_MOTORCYCLE;
    else if (vehPars.obstacleType == "pedestrian")
        return ObstacleData::OBSTACLE_PEDESTRIAN;
    else if (vehPars.obstacleType == "special")
        return ObstacleData::OBSTACLE_SPECIAL;
    else
        return ObstacleData::OBSTACLE_UNKNOWN;
}

/**
 * Check for an upcoming pedestrian crosswalk and navigate through it
 * If no pedestrians are waiting for or occupying the crosswalk, drive through it
 * Otherwise, wait for it to be vacant
 */
void AgentVehicle::checkForCrosswalk(double dt)
{
    //fprintf(stderr,"checkForCrosswalk %d\n",currentCrosswalk);
    double lookoutDist = vehPars.dUtarget;
    if (currentCrosswalk != NULL && crossId != Crosswalk::DONOTENTER)
    {
        // Currently approaching/driving through a crosswalk, check whether it's been crossed already
        if (!crossing)
        {
            if (currentTransition->road->getCrosswalk(u, currentLane) == currentCrosswalk)
            {
                // Have entered the crosswalk
                crossing = true;
            }
        }
        else
        {
            if (currentTransition->road->getCrosswalk(u, currentLane) != currentCrosswalk)
            {
                // Have left the crosswalk
                currentCrosswalk->exitVehicle(crossId, opencover::cover->frameTime());
                crossing = false;
                canBePassed = true;
                currentCrosswalk = NULL;
                crossId = Crosswalk::DONOTENTER;
            }
        }
    }
    else if (currentTransition->road->isAnyCrosswalks(u, (currentTransition->direction > 0 ? u + lookoutDist : u - lookoutDist), currentLane))
    {
        // There's a crosswalk ahead, find it
        Crosswalk *newCrosswalk = currentTransition->road->getNextCrosswalk(u, currentTransition->direction, currentLane);
        if (newCrosswalk == NULL)
            currentCrosswalk = NULL;
        else if (newCrosswalk != currentCrosswalk)
        {
            // Approaching a new crosswalk
            crossPollTimer = 0.0;
            currentCrosswalk = newCrosswalk;
            crossing = false;
            crossId = Crosswalk::DONOTENTER;

            // Attempt to drive through the crosswalk
            crossId = currentCrosswalk->enterVehicle(opencover::cover->frameTime());
            if (crossId == Crosswalk::DONOTENTER)
            {
                // If not possible, stop and wait for crosswalk to be empty
                u -= currentTransition->direction * du * dt;
                du = 0;
                currentCrosswalk = NULL; // try again next time
                canBePassed = false; // don't allow vehicles to pass
            }
            else
            {
                canBePassed = false;
            }
        }
    }
}
