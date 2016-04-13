/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "EinspurDynamikRealtime.h"
#include <iostream>
#include <cmath>

#include "SteeringWheel.h"

#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include <osgUtil/IntersectVisitor>

#include "VehicleUtil.h"

#include <deque>

EinspurDynamikRealtime::EinspurDynamikRealtime()
#ifdef MERCURY
    : XenomaiTask::XenomaiTask("EinspurDynamikRealtimeTask", 0, 99, 0)
#else
    : XenomaiTask::XenomaiTask("EinspurDynamikRealtimeTask", 0, 99, T_FPU | T_CPU(1))
#endif
{
    m = coCoviseConfig::getFloat("mass", "COVER.Plugin.SteeringWheel.EinspurDynamik.inertia", 1500);
    I = coCoviseConfig::getFloat("moiYaw", "COVER.Plugin.SteeringWheel.EinspurDynamik.inertia", 2700);
    lv = coCoviseConfig::getFloat("lengthfront", "COVER.Plugin.SteeringWheel.EinspurDynamik.measures", 2.5);
    lh = coCoviseConfig::getFloat("lengthrear", "COVER.Plugin.SteeringWheel.EinspurDynamik.measures", 2.3);

    roadFrictionForce = 0.0;
    yNull.x = 0;
    yNull.y = 0;
    yNull.kappa = 0;
    yNull.v = 0;
    yNull.alpha = 0;
    yNull.epsilon = 0;
    yNull.dEpsilon = 0;
    yNull.zeta = 0;
    yNull.dZeta = 0;
    yNull.u = 0;

    stepcount = 0;

    errorcontrol = coCoviseConfig::getFloat("error", "COVER.Plugin.SteeringWheel.Dynamics", 1e-4);
    num_intsteps = 1;

    i[0] = coCoviseConfig::getFloat("reverse", "COVER.Plugin.SteeringWheel.Dynamics.transmission", -3.6);
    i[1] = 0;
    i[2] = coCoviseConfig::getFloat("first", "COVER.Plugin.SteeringWheel.Dynamics.transmission", 3.6);
    i[3] = coCoviseConfig::getFloat("second", "COVER.Plugin.SteeringWheel.Dynamics.transmission", 2.19);
    i[4] = coCoviseConfig::getFloat("third", "COVER.Plugin.SteeringWheel.Dynamics.transmission", 1.41);
    i[5] = coCoviseConfig::getFloat("fourth", "COVER.Plugin.SteeringWheel.Dynamics.transmission", 1);
    i[6] = coCoviseConfig::getFloat("fifth", "COVER.Plugin.SteeringWheel.Dynamics.transmission", 0.83);
    iag = coCoviseConfig::getFloat("axle", "COVER.Plugin.SteeringWheel.Dynamics.transmission", 3.5);

    b = coCoviseConfig::getFloat("width", "COVER.Plugin.SteeringWheel.Dynamics.measures", 1.8);
    h = coCoviseConfig::getFloat("height", "COVER.Plugin.SteeringWheel.Dynamics.measures", 1.3);
    Aw = coCoviseConfig::getFloat("Aw", "COVER.Plugin.SteeringWheel.Dynamics.aerodynamics", 2.3);
    cw = coCoviseConfig::getFloat("cw", "COVER.Plugin.SteeringWheel.Dynamics.aerodynamics", 0.3);

    hs = coCoviseConfig::getFloat("heightcenter", "COVER.Plugin.SteeringWheel.Dynamics.inertia", 0.5);
    Iw = coCoviseConfig::getFloat("moiRoll", "COVER.Plugin.SteeringWheel.Dynamics.inertia", 600);
    kw = coCoviseConfig::getFloat("kr", "COVER.Plugin.SteeringWheel.Dynamics.roll", 600000);
    dw = coCoviseConfig::getFloat("dr", "COVER.Plugin.SteeringWheel.Dynamics.roll", 20000);

    In = coCoviseConfig::getFloat("moiPitch", "COVER.Plugin.SteeringWheel.Dynamics.inertia", 2800);
    kn = coCoviseConfig::getFloat("kp", "COVER.Plugin.SteeringWheel.Dynamics.pitch", 200000);
    dn = coCoviseConfig::getFloat("dp", "COVER.Plugin.SteeringWheel.Dynamics.pitch", 50000);

    Im = coCoviseConfig::getFloat("moiMotor", "COVER.Plugin.SteeringWheel.Dynamics.engine", 0.5);
    km = coCoviseConfig::getFloat("km", "COVER.Plugin.SteeringWheel.Dynamics.engine", 200);
    cm = coCoviseConfig::getFloat("cm", "COVER.Plugin.SteeringWheel.Dynamics.engine", 0.01);

    kc = coCoviseConfig::getFloat("kc", "COVER.Plugin.SteeringWheel.Dynamics.transmission", 800);
    cc = coCoviseConfig::getFloat("cc", "COVER.Plugin.SteeringWheel.Dynamics.transmission", 0.04);

    wr = coCoviseConfig::getFloat("wheelradius", "COVER.Plugin.SteeringWheel.Dynamics.measures", 0.25);

    chassisTrans.makeIdentity();
    bodyTrans.makeIdentity();

    accPedal = 0;
    brakePedal = 0;
    clutchPedal = 0;
    gear = 0;
    steerWheelAngle = 0;
    steerPosition = 0;
    steerSpeed = 0;

    double Mm = coCoviseConfig::getFloat("maxoutputmoment", "COVER.Plugin.SteeringWheel.Dynamics.engine", 400);
    double um = 2.0 * M_PI / 60.0 * coCoviseConfig::getFloat("maxoutputspeed", "COVER.Plugin.SteeringWheel.Dynamics.engine", 4600);
    ul = 2.0 * M_PI / 60.0 * coCoviseConfig::getFloat("speedlimit", "COVER.Plugin.SteeringWheel.Dynamics.engine", 7300);
    ui = 2.0 * M_PI / 60.0 * coCoviseConfig::getFloat("idlespeed", "COVER.Plugin.SteeringWheel.Dynamics.engine", 800);
    if (ul > 2 * um)
    {
        ul = 2 * um;
        std::cerr << "Engine speed limit to high for motor characteristic, setting to " << 60.0 / (2.0 * M_PI) * ul << std::endl;
    }
    ach = -km / (um * um);
    bch = (2 * km) / um;
    cch = Mm;

    //accPedalIdle = (km*tanh(cm*ui))/(ach * ui * ui + bch * ui + cch);
    accPedalIdle = 0.1;

    runTask = true;
    taskFinished = false;
    returningToAction = false;
    movingToGround = false;
    pause = true;
    overruns = 0;
    if (coVRMSController::instance()->isMaster())
    {
        //motPlat = new ValidateMotionPlatform("rtcan0");
        motPlat = ValidateMotionPlatform::instance();

        steerCon = new CanOpenController("rtcan1");
        steerWheel = new XenomaiSteeringWheel(*steerCon, 1);

        start();
    }
}

EinspurDynamikRealtime::~EinspurDynamikRealtime()
{
    if (coVRMSController::instance()->isMaster())
    {
        RT_TASK_INFO info;
        inquire(info);

#ifdef MERCURY
    if (info.stat.status & __THREAD_S_STARTED)
#else
    if (info.status & T_STARTED)
#endif
        {
            runTask = false;
            while (!taskFinished)
            {
                usleep(100000);
            }
        }

        delete steerWheel;
        delete steerCon;

        delete motPlat;
    }
}
double EinspurDynamikRealtime::getEngineSpeed()
{
    if (coVRMSController::instance()->isMaster())
    {
        if (VehicleUtil::instance()->getVehicleState() == VehicleUtil::KEYIN_ERUNNING)
        {
            if (pause)
            {
                return ui / (2 * M_PI);
            }
            return yOne.u / (2 * M_PI);
        }
    }
    return 0.0;
}

void EinspurDynamikRealtime::setRoadType(int r)
{
    VehicleDynamics::setRoadType(r);
    if (r == 1)
        roadFrictionForce = 20000.0;
    else
        roadFrictionForce = 0.0;
}

void EinspurDynamikRealtime::setFactors()
{
    accPedal = InputDevice::instance()->getAccelerationPedal();
    if (accPedal < 0.0)
        accPedal = 0.0;
    else if (accPedal > 1.0)
        accPedal = 1.0;
    //brakePedal = InputDevice::instance()->getBrakePedal();
    //if(brakePedal<0.0) brakePedal = 0.0;
    //else if(brakePedal>1.0) brakePedal = 1.0;
    clutchPedal = InputDevice::instance()->getClutchPedal();
    gear = InputDevice::instance()->getGear();
    //steerWheelAngle = InputDevice::instance()->getSteeringWheelAngle();
    steerWheelAngle = -2 * M_PI * ((double)steerPosition / (double)steerWheel->countsPerTurn);

    if (gear == 0)
        clutchPedal = 1;

    this->beta = 0.125 * steerWheelAngle;

    q = (lv + lh) / sin(beta);
    s = (lv + lh) / tan(beta);
    if (beta < 0)
    {
        r = -sqrt(lh * lh + s * s);
        rv = -sqrt(pow((lv + lh) * 0.5, 2) + s * s);
    }
    else
    {
        r = sqrt(lh * lh + s * s);
        rv = sqrt(pow((lv + lh) * 0.5, 2) + s * s);
    }

    gamma = asin(lh / r);
    if (gamma != gamma)
        gamma = 0;

    gammav = asin((lv + lh) * 0.5 / rv);
    if (gammav != gammav)
        gammav = 0;
}

void EinspurDynamikRealtime::setInitialState(double x, double y, double v, double alpha)
{
    yNull.x = x;
    yNull.y = y;
    yNull.v = v;
    yNull.alpha = alpha;
}

void EinspurDynamikRealtime::integrate(double dT, double /*x*/, double /*y*/, double /*alpha*/)
{
    if (dT > 0.1)
        dT = 0.1; //Hardlimit for big times! (We can't handle that.)

    /*yNull.x = x;
	yNull.y = y;
	yNull.alpha = alpha;*/

    setFactors();

    if (num_intsteps < 1)
        num_intsteps = 1;

    double h = dT / num_intsteps;
    //std::cerr << "h: " << h << "dT: " << dT << "num_intsteps: " << num_intsteps << std::endl;

    double s = 0;
    for (int i = 0; i < num_intsteps; ++i)
    {

        dYOne = dFunction(yNull);
        EinspurDynamikRealtimeState k1 = dYOne * h;
        EinspurDynamikRealtimeState k2 = dFunction(yNull + k1 * (1.0 / 4.0)) * h;
        EinspurDynamikRealtimeState k3 = dFunction(yNull + k1 * (3.0 / 32.0) + k2 * (9.0 / 32.0)) * h;
        EinspurDynamikRealtimeState k4 = dFunction(yNull + k1 * (1932.0 / 2197.0) + k2 * (-7200.0 / 2197.0) + k3 * (7296.0 / 2197.0)) * h;
        EinspurDynamikRealtimeState k5 = dFunction(yNull + k1 * (439.0 / 216.0) + k2 * (-8.0) + k3 * (3680.0 / 513.0) + k4 * (-845.0 / 4104.0)) * h;
        EinspurDynamikRealtimeState k6 = dFunction(yNull + k1 * (-8.0 / 27.0) + k2 * (2.0) + k3 * (-3544.0 / 2565.0) + k4 * (1859.0 / 4104.0) + k5 * (-11.0 / 40.0)) * h;

        EinspurDynamikRealtimeState zOne = yNull + k1 * (25.0 / 216.0) + k3 * (1408.0 / 2565.0) + k4 * (2197.0 / 4104.0) - k5 * (1.0 / 5.0);
        yOne = yNull + k1 * (16.0 / 135.0) + k3 * (6656.0 / 12825.0) + k4 * (28561.0 / 56430.0) - k5 * (9.0 / 50.0) + k6 * (2.0 / 55.0);

        s += pow(errorcontrol * h / (2 * norm(yOne - zOne)), 0.25);
        /*
	if((stepcount % 10)==0) {
		std::cerr << "Integration Step:" << std::endl;
		std::cerr << "x: " << yNull.x << ", y: " << yNull.y << ", alpha: " << yNull.alpha << ", v: " << yNull.v << std::endl;
	   std::cerr << "x: " << yOne.x << ", y: " << yOne.y << ", alpha: " << yOne.alpha << ", v: " << yOne.v << std::endl;
	}
	*/
        if (yOne.u < 0)
        {
            std::cerr << "!!!u negative: " << yOne.u << "!!!";
            std::cerr << " Setting u to zero!!!" << std::endl;
            yOne.u = 0.0;
        }

        yNull = yOne;

        ++stepcount;
    }
    num_intsteps = (int)ceil(dT / s);
    //std::cerr << "s: " << s << ", h: " << h << ", num_intsteps: " << num_intsteps << std::endl;

    //std::cerr << "v: " << yOne.v << ", omegav: " << (this->iag * this->i[gear+1] * yOne.v / wr) << ", u: " << yOne.u << std::endl;
}

void EinspurDynamikRealtime::setVehicleTransformation(const osg::Matrix &m)
{
    chassisTrans = m;
}

EinspurDynamikRealtimeDState EinspurDynamikRealtime::dFunction(EinspurDynamikRealtimeState y)
{
    EinspurDynamikRealtimeDState dy;

    dy.dX = y.v * cos(y.alpha + gammav);
    dy.dY = y.v * sin(y.alpha + gammav);

    dy.dKappa = y.v / (2 * M_PI * wr);

    dy.dAlpha = y.v / rv;

    double omegas = y.u - iag * i[gear + 1] * y.v / wr;
    double Mc = kc * (1 - clutchPedal) * tanh(cc * omegas);
    //std::cerr << "omegas: " << omegas << ", Mc: " << Mc << ", u:" << y.u << ", v:" << y.v << std::endl;

    //if(gear == 0) brakePedal=accPedal;    /// Porsche HACK!!!
    double Fa = iag * i[gear + 1] * Mc / wr + (-brakePedal * 20000) * tanh(5 * y.v) - roadFrictionForce * tanh(0.1 * y.v) - 0.65 * cw * Aw * y.v * y.v;

    double dv = (Fa * r * (lv - lh) * cos(beta)) / (m * r * lh * sin(beta) * sin(gamma) + m * r * (lv - lh) * cos(beta) * cos(gamma) + I * sin(beta));
    if (dv != dv)
        dv = Fa / m;

    dy.dV = dv;
    a = dv;

    double Mm;
    //Mm = -0.001 * y.u * y.u + 0.91 * y.u + 400;				//Motorkennlinie
    if (y.u > ul)
        Mm = 0; // Engine speed limit
    else
        Mm = ach * y.u * y.u + bch * y.u + cch; //engine characteristics

    if ((y.u < ui) && (y.u >= 0.0))
        Mm += (ui - y.u) * 0.1; // Extra start aiding torque

    if (accPedal < accPedalIdle) //Leerlaufgas
        accPedal = accPedalIdle;
    Mm *= accPedal;

    Mm += -km * tanh(cm * y.u); //Motorbremsmoment

    dy.dU = 1 / Im * (Mm - Mc);

    //if((stepcount % 10)==0)
    //	std::cerr   << "Mm: " << Mm << ", Fa: " << Fa << ", Mc: " << Mc << ", u: " << y.u/(2*M_PI)*60
    //   << ", ui:" << ui/(2*M_PI)*60
    //            << ", accPedal: " << accPedal << ", accPedalIdle: " << accPedalIdle << std::endl;

    double Mw = -hs / r * m * y.v * y.v;
    if (Mw != Mw)
        Mw = 0;
    dy.dEpsilon = y.dEpsilon;
    dy.ddEpsilon = 1 / Iw * (Mw - dw * y.dEpsilon - kw * y.epsilon);

    double Mn = dv * m * hs;
    dy.dZeta = y.dZeta;
    dy.ddZeta = 1 / In * (Mn - dn * y.dZeta - kn * y.zeta);

    return dy;
}

EinspurDynamikRealtimeState EinspurDynamikRealtimeState::operator+(const EinspurDynamikRealtimeState &state) const
{
    EinspurDynamikRealtimeState result;
    result.x = (this->x + state.x);
    result.y = (this->y + state.y);
    result.kappa = (this->kappa + state.kappa);
    result.alpha = (this->alpha + state.alpha);
    result.v = (this->v + state.v);
    result.u = (this->u + state.u);
    result.epsilon = (this->epsilon + state.epsilon);
    result.dEpsilon = (this->dEpsilon + state.dEpsilon);
    result.zeta = (this->zeta + state.zeta);
    result.dZeta = (this->dZeta + state.dZeta);
    return result;
}

EinspurDynamikRealtimeState EinspurDynamikRealtimeState::operator-(const EinspurDynamikRealtimeState &state) const
{
    EinspurDynamikRealtimeState result;
    result.x = (this->x - state.x);
    result.y = (this->y - state.y);
    result.kappa = (this->kappa - state.kappa);
    result.alpha = (this->alpha - state.alpha);
    result.v = (this->v - state.v);
    result.u = (this->u - state.u);
    result.epsilon = (this->epsilon - state.epsilon);
    result.dEpsilon = (this->dEpsilon - state.dEpsilon);
    result.zeta = (this->zeta - state.zeta);
    result.dZeta = (this->dZeta - state.dZeta);
    return result;
}

EinspurDynamikRealtimeState &EinspurDynamikRealtimeState::operator=(const EinspurDynamikRealtimeState &state)
{
    this->x = state.x;
    this->y = state.y;
    this->kappa = state.kappa;
    this->alpha = state.alpha;
    this->v = state.v;
    this->u = state.u;
    this->epsilon = state.epsilon;
    this->dEpsilon = state.dEpsilon;
    this->zeta = state.zeta;
    this->dZeta = state.dZeta;

    return *this;
}

EinspurDynamikRealtimeState EinspurDynamikRealtimeState::operator*(const double scalar) const
{
    EinspurDynamikRealtimeState result;
    result.x = this->x * scalar;
    result.y = this->y * scalar;
    result.kappa = this->kappa * scalar;
    result.alpha = this->alpha * scalar;
    result.v = this->v * scalar;
    result.u = this->u * scalar;
    result.epsilon = this->epsilon * scalar;
    result.dEpsilon = this->dEpsilon * scalar;
    result.zeta = this->zeta * scalar;
    result.dZeta = this->dZeta * scalar;

    return result;
}

EinspurDynamikRealtimeDState EinspurDynamikRealtimeDState::operator+(const EinspurDynamikRealtimeDState &state) const
{
    EinspurDynamikRealtimeDState result;
    result.dX = (this->dX + state.dX);
    result.dY = (this->dY + state.dY);
    result.dKappa = (this->dKappa + state.dKappa);
    result.dAlpha = (this->dAlpha + state.dAlpha);
    result.dV = (this->dV + state.dV);
    result.dU = (this->dU + state.dU);
    result.dEpsilon = (this->dEpsilon + state.dEpsilon);
    result.ddEpsilon = (this->ddEpsilon + state.ddEpsilon);
    result.dZeta = (this->dZeta + state.dZeta);
    result.ddZeta = (this->ddZeta + state.ddZeta);
    return result;
}

EinspurDynamikRealtimeDState &EinspurDynamikRealtimeDState::operator=(const EinspurDynamikRealtimeDState &state)
{
    this->dX = state.dX;
    this->dY = state.dY;
    this->dKappa = state.dKappa;
    this->dAlpha = state.dAlpha;
    this->dV = state.dV;
    this->dU = state.dU;
    this->dEpsilon = state.dEpsilon;
    this->ddEpsilon = state.ddEpsilon;
    this->dZeta = state.dZeta;
    this->ddZeta = state.ddZeta;

    return *this;
}

EinspurDynamikRealtimeState EinspurDynamikRealtimeDState::operator*(const double scalar) const
{
    EinspurDynamikRealtimeState result;
    result.x = this->dX * scalar;
    result.y = this->dY * scalar;
    result.kappa = this->dKappa * scalar;
    result.alpha = this->dAlpha * scalar;
    result.v = this->dV * scalar;
    result.u = this->dU * scalar;
    result.epsilon = this->dEpsilon * scalar;
    result.dEpsilon = this->ddEpsilon * scalar;
    result.zeta = this->dZeta * scalar;
    result.dZeta = this->ddZeta * scalar;

    return result;
}

double EinspurDynamikRealtime::norm(const EinspurDynamikRealtimeState &state)
{
    return sqrt(state.x * state.x + state.y * state.y + state.kappa * state.kappa + state.alpha * state.alpha + state.v * state.v + state.u * state.u + state.epsilon * state.epsilon + state.dEpsilon * state.dEpsilon + state.zeta * state.zeta + state.dZeta * state.dZeta);
}
void EinspurDynamikRealtime::run()
{
    motPlat->start();
    set_periodic(period);

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
    motPlat->getSendMutex().release();
    rt_task_wait_period(&overruns);

    //SteeringWheel
    set_periodic(TM_INFINITE);
    steerWheel->init();
    std::deque<int32_t> steerSpeedDeque(50, 0);
    std::deque<int32_t>::iterator steerSpeedDequeIt;
    double lowPassSpeed = 0.0;
    int32_t driftPosition = 0;

    set_periodic(period);
    while (runTask)
    {
        brakePedal = std::max(0.0, std::min(1.0, ValidateMotionPlatform::instance()->getBrakePedalPosition() * 5.0));

        if (!pause)
        {
            double h = (double)(period * (overruns + 1)) * 1e-9;
            integrate(h, 0, 0, 0);
        }
        /*longAngleDeque.pop_front();
      longAngleDeque.push_back(-yOne.zeta);
      double longAngle=0;
      for(angleDequeIt=longAngleDeque.begin(); angleDequeIt!=longAngleDeque.end(); ++angleDequeIt) {
         longAngle += (*angleDequeIt);
      }
      longAngle = longAngle/(double)longAngleDeque.size();

      latAngleDeque.pop_front();
      latAngleDeque.push_back(-yOne.epsilon);
      double latAngle=0;
      for(angleDequeIt=latAngleDeque.begin(); angleDequeIt!=latAngleDeque.end(); ++angleDequeIt) {
         latAngle += (*angleDequeIt);
      }
      latAngle = latAngle/(double)latAngleDeque.size();*/

        if (movingToGround)
        {
            if (motPlat->isGrounded())
            {
                movingToGround = false;
                motPlat->getSendMutex().acquire(period);
                motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
                motPlat->getSendMutex().release();
            }
        }
        else if (returningToAction)
        {
            if (motPlat->isMiddleLifted())
            {
                returningToAction = false;
                pause = false;

                motPlat->getSendMutex().acquire(period);
                //motPlat->switchToMode<ValidateMotionPlatform::controlPositioning>();
                motPlat->switchToMode<ValidateMotionPlatform::controlInterpolatedPositioning>();
                for (unsigned int motIt = 0; motIt < motPlat->numLinMots; ++motIt)
                {
                    motPlat->setVelocitySetpoint(motIt, ValidateMotionPlatform::velMax);
                    motPlat->setAccelerationSetpoint(motIt, ValidateMotionPlatform::accMax);
                }
                motPlat->getSendMutex().release();
            }
        }
        else if (!pause)
        {
            double longAngle = -yOne.zeta;
            double latAngle = -yOne.epsilon;

            motPlat->getSendMutex().acquire(period);
            motPlat->setLongitudinalAngleSetpoint(longAngle);
            motPlat->setLateralAngleSetpoint(latAngle);
            motPlat->getSendMutex().release();
        }

        //SteeringWheel
        steerCon->sendSync();
        steerCon->recvPDO(1);

        //steerPosition = steerWheel->getPosition();
        //steerSpeed = steerWheel->getSpeed();
        steerWheel->getPositionSpeed(steerPosition, steerSpeed);
        *(((uint8_t *)&steerSpeed) + 3) = (*(((uint8_t *)&steerSpeed) + 2) & 0x80) ? 0xff : 0x0;
        steerSpeedDeque.pop_front();
        steerSpeedDeque.push_back(steerSpeed);
        lowPassSpeed = 0;
        for (steerSpeedDequeIt = steerSpeedDeque.begin(); steerSpeedDequeIt != steerSpeedDeque.end(); ++steerSpeedDequeIt)
        {
            lowPassSpeed += (*steerSpeedDequeIt);
        }
        lowPassSpeed = (double)lowPassSpeed / (double)steerSpeedDeque.size();

        double drillElasticity = tanh(fabs(yOne.v));
        int32_t current = (int32_t)(-steerWheel->getSpringConstant() * drillElasticity * (double)steerPosition - steerWheel->getDampingConstant() * lowPassSpeed); //Spring-Damping-Model

        current += (int32_t)(((rand() / ((double)RAND_MAX)) - 0.5) * steerWheel->getRumbleAmplitude()); //Rumbling

        double drillRigidness = 1.0 - drillElasticity;
        if ((driftPosition - steerPosition) > 100000 * drillRigidness)
            driftPosition = steerPosition + (int32_t)(100000 * drillRigidness);
        else if ((driftPosition - steerPosition) < -100000 * drillRigidness)
            driftPosition = steerPosition - (int32_t)(100000 * drillRigidness);
        current += (int32_t)((double)(driftPosition - steerPosition) * steerWheel->getDrillConstant());
        //std::cerr << "drift steerPosition - steerPosition: " << (int32_t)((double)(driftPosition-steerPosition)*0.005) << ", drill current: " << (int32_t)((double)(driftPosition-steerPosition)*Kdrill) << ", Kdrill: " << Kdrill << std::endl;
        steerWheel->setCurrent(current);

        steerCon->sendPDO();

        /*if((count % 1000)==0) {
         std::cerr << "Count: " << count << ", overruns: " << overruns << std::endl;
      }
      ++count;*/
        if (overruns > 0)
        {
            std::cerr << "EinspurDynamikRealtime::run(): overruns: " << overruns << std::endl;
        }
        rt_task_wait_period(&overruns);
    }
    /*steerCon->sendSync();
   rt_task_wait_period(&overruns);
   rt_task_wait_period(&overruns);
   steerCon->sendSync();*/

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
    motPlat->getSendMutex().release();
    while (!motPlat->isGrounded())
    {
        rt_task_wait_period(&overruns);
    }
    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlDisabled>();
    motPlat->getSendMutex().release();

    set_periodic(TM_INFINITE);
    //SteerWheel
    //rt_task_sleep(10000000);
    steerWheel->shutdown();

    taskFinished = true;
}

void EinspurDynamikRealtime::platformToGround()
{
    pause = true;
    movingToGround = true;
    returningToAction = false;
    memset(&yOne, 0, sizeof(yOne));
    memset(&yNull, 0, sizeof(yNull));

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlToGround>();
    motPlat->getSendMutex().release();
}

void EinspurDynamikRealtime::platformReturnToAction()
{
    pause = true;
    returningToAction = true;
    movingToGround = false;

    motPlat->getSendMutex().acquire(period);
    motPlat->switchToMode<ValidateMotionPlatform::controlMiddleLift>();
    motPlat->getSendMutex().release();
}

void EinspurDynamikRealtime::move(VrmlNodeVehicle *vehicle)
{
    EinspurDynamikRealtimeState yOne;
    if (coVRMSController::instance()->isMaster())
    {
        yOne = this->yOne;
        coVRMSController::instance()->sendSlaves((char *)&yOne, sizeof(yOne));
        yNull.x = 0;
        yNull.y = 0;
        yNull.alpha = 0;
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&yOne, sizeof(yOne));
    }

    //double clutchPedal = InputDevice::getInstance()->getClutchPedal();
    //bool hornButton = InputDevice::getInstance()->getHornButton();
    //bool resetButton = InputDevice::getInstance()->getResetButton();

    //v = SteeringWheelPlugin::plugin->einspur->getV();
    //a = SteeringWheelPlugin::plugin->einspur->getA();
    //enginespeed = SteeringWheelPlugin::plugin->einspur->getEngineSpeed();

    osg::Matrix relTrans;
    relTrans.makeTranslate(-yOne.y, 0, -yOne.x);
    osg::Matrix relRotYaw;
    relRotYaw.makeRotate(yOne.alpha, 0, 1, 0);
    osg::Matrix relRotRoll;
    relRotRoll.makeRotate(yOne.epsilon, 0, 0, 1);
    osg::Matrix relRotPitch;
    relRotPitch.makeRotate(yOne.zeta, 1, 0, 0);

    chassisTrans = relRotYaw * relTrans * chassisTrans;
    bodyTrans = relRotPitch * relRotRoll;

    vehicle->moveToStreet(chassisTrans);
    vehicle->setVRMLVehicle(chassisTrans);
    vehicle->setVRMLVehicleBody(bodyTrans);

    osg::Matrix frontWheelTrans;
    frontWheelTrans.makeIdentity();
    osg::Matrix rearWheelTrans;
    rearWheelTrans.makeIdentity();
    osg::Quat wheelRevs;
    osg::Quat wheelBeta;
    osg::Quat frontWheelRotation;

    wheelBeta.makeRotate(beta, 0, 0, 1);
    wheelRevs.makeRotate(yOne.kappa, 0, -1, 0);
    frontWheelRotation = wheelRevs * wheelBeta;

    frontWheelTrans.setRotate(frontWheelRotation);
    vehicle->setVRMLVehicleFrontWheels(frontWheelTrans, frontWheelTrans);

    rearWheelTrans.setRotate(wheelRevs);
    vehicle->setVRMLVehicleRearWheels(rearWheelTrans, rearWheelTrans);

    /*double elevOne = 0;
   double elevTwo = 0;
   double elevThree = 10;
   double elevFour = 10;

   osg::Vec3 vehPos = vehicle->getCarTransMatrix()->getTrans();
   osg::Vec2d wheelPos(vehPos.x(), -vehPos.z());

   getWheelElevation(wheelPos, wheelPos, wheelPos, wheelPos, elevOne, elevTwo, elevThree, elevFour);
   std::cout << "Elevation pos: " << wheelPos[0] << ", " << wheelPos[1] << ": one: " << elevOne << ", two: " << elevTwo << ", three: " << elevThree << ", four: " << elevFour << std::endl;*/
}

void EinspurDynamikRealtime::resetState()
{
    yNull.x = 0;
    yNull.y = 0;
    yNull.kappa = 0;
    yNull.v = 0;
    yNull.u = 0;
    yNull.alpha = 0;
    yNull.epsilon = 0;
    yNull.dEpsilon = 0;
    yNull.zeta = 0;
    yNull.dZeta = 0;

    chassisTrans.makeIdentity();
    bodyTrans.makeIdentity();
}
