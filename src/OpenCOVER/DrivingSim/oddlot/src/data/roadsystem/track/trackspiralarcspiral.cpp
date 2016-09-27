/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   08.04.2010
**
**************************************************************************/

#include "trackspiralarcspiral.hpp"
#include <cmath>
#include <float.h>

#include "trackelementarc.hpp"
#include "trackelementspiral.hpp"

#include "src/data/roadsystem/rsystemelementroad.hpp"

// utils //
//
#include <QDebug>
#include <QVector2D>
#include <QTransform>

#include <util/unixcompat.h>

//################//
// CONSTRUCTOR    //
//################//

TrackSpiralArcSpiral::TrackSpiralArcSpiral(TrackElementSpiral *inSpiral, TrackElementArc *arc, TrackElementSpiral *outSpiral)
    : TrackComposite()
    , inSpiral_(inSpiral)
    , arc_(arc)
    , outSpiral_(outSpiral)
    , trackSpArcSChanges_(0x0)
    , validParameters_(false)
{
    // Type //
    //
    setTrackType(TrackComponent::DTT_SPARCS);

    // Transform //
    //
    double sHeading = inSpiral_->getLocalHeading(inSpiral_->getSStart());
    setLocalTransform(inSpiral_->getLocalPoint(inSpiral_->getSStart()), sHeading);

    inSpiral_->setLocalTransform(0.0, 0.0, 0.0); // identity matrix
    arc_->setLocalTransform(getLocalTransform().inverted().map(arc_->getLocalPoint(arc_->getSStart())), arc_->getLocalHeading(arc_->getSStart()) - sHeading);
    outSpiral_->setLocalTransform(getLocalTransform().inverted().map(outSpiral_->getLocalPoint(outSpiral_->getSStart())), outSpiral_->getLocalHeading(outSpiral_->getSStart()) - sHeading);

    // Composite //
    //
    addTrackComponent(inSpiral);
    addTrackComponent(arc);
    addTrackComponent(outSpiral);

    // Parameters //
    //
    pa_ = new SpArcSParameters(outSpiral_->getLocalPoint(outSpiral_->getSEnd()), outSpiral_->getLocalHeadingRad(outSpiral_->getSEnd()), 1.0);

    pa_->setFactor(calcFactor());

    if (!pa_->isValid()) // does also initialize the parameters
    {
        validParameters_ = false;
        qDebug() << "WARNING 1004261530! Loaded TrackSpiralArcSpiral is not valid!";
    }
    else
    {
        validParameters_ = true;
    }
}

TrackSpiralArcSpiral::TrackSpiralArcSpiral(const QPointF &startPos, const QPointF &endPos, double startHeadingDeg, double endHeadingDeg, double factor)
    : TrackComposite()
    , inSpiral_(NULL)
    , arc_(NULL)
    , outSpiral_(NULL)
    , trackSpArcSChanges_(0x0)
    , validParameters_(false)
{
    // Type //
    //
    setTrackType(TrackComponent::DTT_SPARCS);

    // Track //
    //
    inSpiral_ = new TrackElementSpiral(0.0, 0.0, 0.0, 0.0, 49.6551724137931, 0.0, 0.003448275862068966);
    arc_ = new TrackElementArc(49.61879011646819, 1.416290576997416, 0.0856123662306776 * 360.0 / (2.0 * M_PI), 49.65517241379311, 124.3448275876508, 0.003448275862068966);
    outSpiral_ = new TrackElementSpiral(167.5020491350196, 37.88185581403775, 0.5143876337743012 * 360.0 / (2.0 * M_PI), 174, 49.6551724137931, 0.003448275862068966, 0.0);

    addTrackComponent(inSpiral_);
    addTrackComponent(arc_);
    addTrackComponent(outSpiral_);

    setLocalTransform(startPos, startHeadingDeg);

    // Parameters //
    //
    pa_ = new SpArcSParameters(outSpiral_->getLocalPoint(outSpiral_->getSEnd()), outSpiral_->getLocalHeadingRad(outSpiral_->getSEnd()), factor);
    //	pa_->setFactor(calcFactor());

    setGlobalPointAndHeading(endPos, endHeadingDeg, false);

    if (pa_->isValid())
    {
        validParameters_ = true;
    }
}

TrackSpiralArcSpiral::~TrackSpiralArcSpiral()
{
    delete pa_;
}

//###################//
// SpArcS            //
//###################//

/*! \brief Returns the parameters (cloned).
*
*/
SpArcSParameters *
TrackSpiralArcSpiral::getClonedParameters() const
{
    return new SpArcSParameters(pa_->pEnd_, pa_->headingEnd_, pa_->factor_);
}

/*! \brief Convenience function.
*
*/
void
TrackSpiralArcSpiral::setLocalStartPoint(const QPointF &startPoint)
{
    // Local to internal (Parameters are given in internal coordinates) //
    //
    QPointF deltaPos(getLocalTransform().inverted().map(startPoint) /* - getPoint(getSStart())*/); // getPoint(s_) == 0 by definition
    //setEndPoint(pa_->pEnd_ - deltaPos);
    pa_->pEnd_ = pa_->pEnd_ - deltaPos;
    pa_->init();
    applyParameters();

    // Set local translation //
    //
    setLocalTranslation(startPoint);
}

/*! \brief Convenience function.
*
*/
void
TrackSpiralArcSpiral::setLocalEndPoint(const QPointF &endPoint)
{
    // Local to internal (Parameters are given in internal coordinates) //
    //
    //	setEndPoint(getLocalTransform().inverted().map(endPoint));
    pa_->pEnd_ = getLocalTransform().inverted().map(endPoint);
    pa_->init();
    applyParameters();
}

/*! \brief Convenience function.
*
*/
void
TrackSpiralArcSpiral::setLocalStartHeading(double startHeading)
{
    while (startHeading <= -180.0)
    {
        startHeading += 360.0;
    }
    while (startHeading > 180.0)
    {
        startHeading -= 360.0;
    }

    // Local to internal (Parameters are given in internal coordinates) //
    //
    double deltaHeading(startHeading - heading());

    QTransform trafo;
    trafo.rotate(deltaHeading);

    pa_->setEndHeadingDeg(pa_->getEndHeadingRad() * 360.0 / (2.0 * M_PI) - deltaHeading);
    pa_->setEndPoint(trafo.inverted().map(pa_->getEndPoint()));
    pa_->init();
    applyParameters();

    // Set local translation //
    //
    setLocalRotation(startHeading);
}

/*! \brief Convenience function.
*
*/
void
TrackSpiralArcSpiral::setLocalEndHeading(double endHeading)
{
    while (endHeading <= -180.0)
    {
        endHeading += 360.0;
    }
    while (endHeading > 180.0)
    {
        endHeading -= 360.0;
    }

    // Local to internal (Parameters are given in internal coordinates) //
    //
    setEndHeadingDeg(endHeading - getLocalHeading(getSStart()));
}

/*! \brief Convenience function.
*
*/
void
TrackSpiralArcSpiral::setLocalPointAndHeading(const QPointF &point, double hdg, bool isStart)
{
    while (hdg <= -180.0)
    {
        hdg += 360.0;
    }
    while (hdg > 180.0)
    {
        hdg -= 360.0;
    }

    if (isStart)
    {
        // Local to internal (Parameters are given in internal coordinates) //
        //
        QPointF deltaPos(getLocalTransform().inverted().map(point) /* - getPoint(getSStart())*/); // getPoint(s_) == 0 by definition
        double deltaHeading(hdg - heading());

        QTransform trafo;
        trafo.rotate(deltaHeading);

        pa_->setEndHeadingDeg(pa_->getEndHeadingRad() * 360.0 / (2.0 * M_PI) - deltaHeading);
        pa_->setEndPoint(trafo.inverted().map(pa_->pEnd_ - deltaPos));
        pa_->init();
        applyParameters();

        // Set local transform //
        //
        setLocalTransform(point, hdg);
    }
    else
    {
        // Local to internal (Parameters are given in internal coordinates) //
        //
        pa_->pEnd_ = getLocalTransform().inverted().map(point);
        pa_->headingEnd_ = (hdg - getLocalHeading(getSStart())) * 2.0 * M_PI / 360.0;

        pa_->init();
        applyParameters();
    }
}

/*! \brief Convenience function.
*
*/
void
TrackSpiralArcSpiral::setEndPoint(const QPointF &endPoint)
{
    pa_->pEnd_ = endPoint;
    pa_->init();
    applyParameters();
}

/*! \brief Convenience function.
*
*/
void
TrackSpiralArcSpiral::setEndHeadingDeg(double endHeadingDeg)
{
    pa_->headingEnd_ = endHeadingDeg * 2.0 * M_PI / 360.0;
    pa_->init();
    applyParameters();
}

/*! \brief Sets the unsymmetric arc insertion factor.
*
*/
void
TrackSpiralArcSpiral::setFactor(double factor)
{
    if (factor < 0.001)
        factor = 0.001;
    if (factor > 0.999)
        factor = 0.999;

    SpArcSParameters *tmp = pa_;
    pa_ = new SpArcSParameters(outSpiral_->getLocalPoint(outSpiral_->getSEnd()), outSpiral_->getLocalHeadingRad(outSpiral_->getSEnd()), factor);
    if (!pa_->isValid()) // does also initialize the parameters
    {
        pa_ = tmp;
        pa_->isValid(); // check again
        qDebug() << "WARNING 1005170517! Factor for TrackSpiralArcSpiral is not valid!";
    }
    else
    {
        delete tmp;
        applyParameters();
    }
}

/*! \brief Calculates the unsymmetric arc insertion factor for the
* given parameters.
*
*/
double
TrackSpiralArcSpiral::calcFactor() const
{
    double tau0 = fabs(inSpiral_->getLength() * inSpiral_->getCurvature(inSpiral_->getSEnd()) * 0.5);
    double tau1 = fabs(outSpiral_->getLength() * outSpiral_->getCurvature(outSpiral_->getSStart()) * 0.5);
    //qDebug() << tau0 << " " << tau1;

    bool success = false;
    double tau0u = pa_->calcTau0u(success);
    double tau1u = pa_->getAngle() - tau0u;
    //qDebug() << tau0u << " " << tau1u;

    if (tau0 < tau1)
    {
        if (tau0u < NUMERICAL_ZERO7)
        {
            return 0.0;
        }
        else
        {
            //qDebug() << "factor: " << tau0/tau0u;
            return tau0 / tau0u;
        }
    }
    else
    {
        if (tau1u < NUMERICAL_ZERO7)
        {
            return 0.0;
        }
        else
        {
            //qDebug() << "factor: " << tau1/tau1u;
            return tau1 / tau1u;
        }
    }
}

/*! \brief Calculates and sets the track parameters.
*
*/
void
TrackSpiralArcSpiral::applyParameters()
{
    if (!pa_->isValid())
    {
        validParameters_ = false;
        //qDebug("WARNING 1006041146! TrackSpiralArcSpiral parameters not valid.");
        return;
    }
    else
    {
        validParameters_ = true;
    }

    // Clothoid Parameters a0, a1 //
    //
    double sqrtTwoTau0 = sqrt(2.0 * pa_->tau0_);
    double sqrtTwoTau1 = sqrt(2.0 * pa_->tau1_);

    double a0 = pa_->h_ * sin(pa_->angle_)
                / (TrackElementSpiral::y(sqrtTwoTau0) + cos(pa_->tau0_) / sqrt(2.0 * pa_->tau0_)
                   + (TrackElementSpiral::x(sqrtTwoTau1) * sin(pa_->angle_) - TrackElementSpiral::y(sqrtTwoTau1) * cos(pa_->angle_)) * sqrt(pa_->tau1_ / pa_->tau0_)
                   - cos(pa_->angle_ - pa_->tau1_) / sqrt(2.0 * pa_->tau0_));
    double a1 = a0 * sqrt(pa_->tau1_ / pa_->tau0_);
    //qDebug() << "a0: " << a0 << ", a1: " << a1;

    // Lengths l0, l1 //
    //
    double l0 = a0 * sqrt(2.0 * pa_->tau0_);
    double l1 = a1 * sqrt(2.0 * pa_->tau1_);
    //qDebug() << "l0: " << l0 << ", l1: " << l1;

    // Curvature //
    //
    //	double c = l0/(a0*a0) * inSpiral_->getAsign();
    double c = l0 / (a0 * a0);
    if (pa_->headingEnd_ < 0.0)
    {
        c = -c; // change sign
    }

    //qDebug() << "c: " << c;

    // Arc //
    //
    double la = (pa_->angle_ - pa_->tau0_ - pa_->tau1_) / fabs(c);

    // InSpiral //
    //
    inSpiral_->setCurvEndAndLength(c, l0);

    // Arc //
    //
    delTrackComponent(arc_);
    double sStartArc = inSpiral_->getSEnd();
    arc_->setSStart(sStartArc);
    arc_->setLength(la);
    arc_->setCurvature(c);
    arc_->setLocalTransform(inSpiral_->getLocalPoint(sStartArc), inSpiral_->getLocalHeading(sStartArc));
    addTrackComponent(arc_);

    // OutSpiral //
    //
    delTrackComponent(outSpiral_);
    double sEndArc = arc_->getSEnd();
    outSpiral_->setSStart(sEndArc);
    outSpiral_->setCurvStartAndLength(c, l1);
    outSpiral_->setLocalTransform(arc_->getLocalPoint(sEndArc), arc_->getLocalHeading(sEndArc));
    addTrackComponent(outSpiral_);

    // Observer Pattern //
    //
    addTrackSpArcSChanges(TrackSpiralArcSpiral::CTV_ParameterChange);

    if (getParentRoad())
    {
        getParentRoad()->rebuildTrackComponentList();
    }
}

//################//
// OBSERVER       //
//################//

/*! \brief Called after all Observers have been notified.
*
* Resets the change flags to 0x0.
*/
void
TrackSpiralArcSpiral::notificationDone()
{
    trackSpArcSChanges_ = 0x0;
    TrackComposite::notificationDone();
}

/*! \brief Add one or more change flags.
*
*/
void
TrackSpiralArcSpiral::addTrackSpArcSChanges(int changes)
{
    if (changes)
    {
        trackSpArcSChanges_ |= changes;
        notifyObservers();
        addTrackComponentChanges(TrackComponent::CTC_LengthChange);
        addTrackComponentChanges(TrackComponent::CTC_ShapeChange);
        addTrackComponentChanges(TrackComponent::CTC_TransformChange);
    }
}

//###################//
// Prototype Pattern //
//###################//

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
TrackComponent *
TrackSpiralArcSpiral::getClone() const
{
    return getClonedSpArcS();
}

/*! \brief Creates and returns a deep copy clone of this object.
*
*/
TrackSpiralArcSpiral *
TrackSpiralArcSpiral::getClonedSpArcS() const
{
    TrackSpiralArcSpiral *sparcs = new TrackSpiralArcSpiral(inSpiral_->getClonedSpiral(), arc_->getClonedArc(), outSpiral_->getClonedSpiral());
    sparcs->setLocalTransform(pos(), heading());
    return sparcs;
}

//#################//
// VISITOR         //
//#################//

/** Accepts a visitor.
*
*/
void
TrackSpiralArcSpiral::accept(Visitor *visitor)
{
    visitor->visit(this);
}

//#################//
//                 //
// PARAMETERS      //
//                 //
//#################//

SpArcSParameters::SpArcSParameters(const QPointF &pEnd, double headingEnd, double factor)
    : pEnd_(pEnd)
    , headingEnd_(headingEnd)
    , factor_(factor)
    , isValid_(-1)
{
    init();
}

void
SpArcSParameters::init()
{
    // Revalidate //
    //
    isValid_ = -1;

    // Angle between tangents //
    //
    while (headingEnd_ <= -M_PI)
    {
        headingEnd_ += 2.0 * M_PI;
    }
    while (headingEnd_ > M_PI)
    {
        headingEnd_ -= 2.0 * M_PI;
    }

    angle_ = fabs(headingEnd_);

    // Calculation of tangent lengths g and h //
    //
    // TODO: optimize
    QVector2D p = QVector2D(pEnd_);

    QVector2D t0 = QVector2D(1.0, 0.0);
    QVector2D t1 = QVector2D(cos(headingEnd_), sin(headingEnd_));
    QVector2D n0 = QVector2D(t0.y(), -t0.x()); // sign doesn't matter here
    QVector2D n1 = QVector2D(t1.y(), -t1.x()); // sign doesn't matter here

    h_ = QVector2D::dotProduct(p, n0) / QVector2D::dotProduct(t1, n0);
    g_ = QVector2D::dotProduct(p, n1) / QVector2D::dotProduct(t0, n1);

    if(h_ < 0.000001)
    {
        k_ = 0;
        h_ = -1; // invalid
    }
    else
    {
        k_ = g_ / h_;
    }
}

/*! \brief Set Parameter.
*/
void
SpArcSParameters::setEndPoint(const QPointF &endPoint)
{
    pEnd_ = endPoint;
    isValid_ = -1;
}

void
SpArcSParameters::setEndHeadingDeg(double endHeadingDeg)
{
    setEndHeadingRad(endHeadingDeg * 2.0 * M_PI / 360.0);
}

void
SpArcSParameters::setEndHeadingRad(double endHeadingRad)
{
    headingEnd_ = endHeadingRad;
    isValid_ = -1;
}

void
SpArcSParameters::setFactor(double factor)
{
    factor_ = factor;
    isValid_ = -1;
}

bool
SpArcSParameters::isValid()
{
    if (isValid_ == 0)
    {
        return false;
    }
    else if (isValid_ == 1)
    {
        return true;
    }
    else // isValid_ == -1
    {
        init();

        if (checkValidity())
        {
            isValid_ = 1;
            return true;
        }
        else
        {
            isValid_ = 0;
            return false;
        }
    }
}

bool
SpArcSParameters::checkValidity()
{
    // GEOMETRY //
    //
    if (g_ < 0 || h_ < 0 || std::isnan(g_) || std::isnan(h_))
    {
        return false;
    }

    // UNSYMMETRIC BLENDING //
    //
    bool success = false;
    double tau0u = calcTau0u(success);
    if (!success)
    {
        return false;
    }

    // FLIP //
    //
    double tau1u = angle_ - tau0u;
    //qDebug() << "Result unsym: " << f(tauApprox) << " " << tauApprox << " in " << iIteration << " steps.";

    bool flipped = (tau0u < tau1u); // so tau0u is always equal or greater than tau1u
    if (flipped)
    {
        double tmp = tau0u;
        tau0u = tau1u;
        tau1u = tmp;
        k_ = 1 / k_;
    }

    // UNSYMMETRIC ARC INSERTION //
    //
    // Insert an arc. Tau1 is given by the factor, tau0 will be calculated.
    double tau1 = tau1u * factor_;

    // First approximation for tau0 //
    //
    double tau0Approx = tau1;
    double tauOld = tau0Approx;
    //qDebug() << "tau1: " << tau1 << "q(tau1)= " << q(tau1, tau1);
    //qDebug() << "angle: " << angle_ << "q(angle-tau1)= " << q(angle_-tau1, tau1);

    double critAI = 0.0001; // stop if improvement is less than ...
    int iIterationAI = 0;
    do
    {
        tauOld = tau0Approx;
        ++iIterationAI;

        // Approximation //
        //
        tau0Approx = tauOld - q(tauOld, tau1) / dq(tauOld);

        // Check if it's ok to continue //
        //
        if (tau0Approx < 0.0)
        {
            //qDebug("TODO, unsymmetric arc insertion: less than zero");
            return false;
        }
        else if (tau0Approx > angle_ - tau1)
        {
            //qDebug("TODO, unsymmetric arc insertion: greater than angle tau1");
            return false;
        }
        else if (iIterationAI >= 50)
        {
            //qDebug("TODO, unsymmetric arc insertion: more than 50 iteration steps");
            return false;
        }
    } while (fabs(tauOld - tau0Approx) > critAI);

    // Save and exit //
    //
    if (flipped)
    {
        //		qDebug("flipped");
        double tmp = tau0Approx;
        tau0Approx = tau1;
        tau1 = tmp;
        k_ = 1 / k_;
    }

    tau1_ = tau1;
    tau0_ = tau0Approx;

    //	qDebug() << "Result: " << tau0Approx << " " << q(tau0Approx, tau1) << " in " << iIterationAI;
    //	qDebug() << tau1 << " < " << tau0Approx << " < " << angle_-tau1u << " = " << tau0u;

    return true;
}

double
SpArcSParameters::calcTau0u(bool &success)
{
    // UNSYMMETRIC BLENDING //
    //
    // See paper "A controlled clothoid spline" by D.J.Walton.
    // Unsymmetric blending is accomplished with two clothoids
    // and no arc. This is the first extremum with tau0 and tau1
    // at their maxima.
    // The other one is tau0 or tau1 equal zero.

    // First approximation //
    //
    double tauApprox = angle_ / 4.0; // 2.0?
    double tauOld = tauApprox;
    //	qDebug() << "Init: " << tauApprox << ", f()=" << f(tauApprox);

    // Newton's Method //
    //
    double crit = 0.000001; // stop if improvement is less than crit
    int iIteration = 0;
    do
    {
        tauOld = tauApprox;
        ++iIteration;

        // Approximation //
        //
        tauApprox = tauOld - f(tauOld) / df(tauOld);

        // Check if it's ok to continue //
        //
        if (k_ == 0 || tauApprox < 0.0)
        {
            //qDebug("TODO, unsymmetric blending: less than zero");
            success = false;
            return 0.0;
        }
        else if (tauApprox > angle_) // actually max is angle_/2.0 but may oscillate around that
        {
            //qDebug("TODO, unsymmetric blending: greater than angle");
            success = false;
            return 0.0;
        }
        else if (iIteration >= 50)
        {
            //qDebug("TODO, unsymmetric blending: more than 50 iteration steps");
            success = false;
            return 0.0;
        }
    } while (fabs(tauOld - tauApprox) > crit);

    // Save and return //
    //
    success = true;
    return tauApprox;
}

double
SpArcSParameters::f(double tau)
{
    double sqrtTwoTau = sqrt(2.0 * tau);
    return sqrt(tau) * (TrackElementSpiral::x(sqrtTwoTau) * sin(angle_) - TrackElementSpiral::y(sqrtTwoTau) * (k_ + cos(angle_)))
           + sqrt(angle_ - tau) * (TrackElementSpiral::y(sqrt(2.0 * (angle_ - tau))) * (1 + k_ * cos(angle_)) - k_ * TrackElementSpiral::x(sqrt(2.0 * (angle_ - tau))) * sin(angle_));
}

double
SpArcSParameters::df(double tau)
{
    double sqrtTwoTau = sqrt(2.0 * tau);
    double CTau = TrackElementSpiral::x(sqrtTwoTau);
    double STau = TrackElementSpiral::y(sqrtTwoTau);
    double CATau = TrackElementSpiral::x(sqrt(2.0 * (angle_ - tau)));
    double SATau = TrackElementSpiral::y(sqrt(2.0 * (angle_ - tau)));

    return 0.5 * STau * sin(angle_) / sqrt(tau)
           * (CTau / STau - (k_ + cos(angle_)) / (sin(angle_)))
           + ((k_ * SATau * sin(angle_)) / (2.0 * sqrt(angle_ - tau)))
             * (CATau / SATau - (1 + k_ * cos(angle_) / (k_ * sin(angle_))));
}

double
SpArcSParameters::q(double tau, double tau1)
{
    double sqrtTwoTau = sqrt(2.0 * tau);
    double sqrtTwoTau1 = sqrt(2.0 * tau1);
    double CTau = TrackElementSpiral::x(sqrtTwoTau);
    double STau = TrackElementSpiral::y(sqrtTwoTau);
    double CTau1 = TrackElementSpiral::x(sqrtTwoTau1);
    double STau1 = TrackElementSpiral::y(sqrtTwoTau1);

    return sqrt(2.0 * tau) * (CTau * sin(angle_) - STau * (cos(angle_) + k_))
           - sqrt(2.0 * tau1) * (k_ * CTau1 * sin(angle_) - STau1 * (k_ * cos(angle_) + 1.0))
           - cos(angle_ - tau) - k_ * cos(tau)
           + cos(tau1) + k_ * cos(angle_ - tau1);
}

double
SpArcSParameters::dq(double tau)
{
    double sqrtTwoTau = sqrt(2.0 * tau);
    double CTau = TrackElementSpiral::x(sqrtTwoTau);
    double STau = TrackElementSpiral::y(sqrtTwoTau);

    return sqrt(0.5 / tau) * STau * sin(angle_) * (CTau / STau - (k_ + cos(angle_)) / sin(angle_));
}

//#################//
// OLD STUFF       //
//#################//

#if 0
	// SYMMETRIC CLOTHOID BLENDING //
	//

	// Distance between inSpiral start and outSpiral end //
	//
	double DistAB = QVector2D(outSpiral_->getLocalPoint(outSpiral_->getSEnd()) - inSpiral_->getLocalPoint(inSpiral_->getSStart())).length();
	qDebug() << "DistAB: " << DistAB;

	// Angle between the tangents (Oeffnungswinkel) //
	//
	double gamma = outSpiral_->getLocalHeadingRad(outSpiral_->getSEnd()) - inSpiral_->getLocalHeadingRad(inSpiral_->getSStart());
	qDebug() << "Gamma: " << gamma;

	//  //
	//
	double T = DistAB / (2.0*cos(fabs(gamma)/2.0));

	double LAratio = sqrt(fabs(gamma));

	double A = T / (TrackElementSpiral::x(LAratio) + TrackElementSpiral::y(LAratio)*tan(fabs(gamma)/2.0));
	qDebug() << "A: " << A;

	double L = A*LAratio;
	qDebug() << "L: " << L;
	qDebug() << "Lsparcs: " << inSpiral_->getLength()+arc_->getLength()/2.0;

	double Curv = L/(A*A);
	qDebug() << "Radius: " << 1/Curv;


//	inSpiral_->setLength(L);
//	inSpiral_->setCurvEnd(-Curv, false);
//
//	delTrackComponent(arc_);
//	double sStartArc = inSpiral_->getSEnd();
//	arc_->setSStart(sStartArc);
//	arc_->setLength(0.001);
//	arc_->setLocalTransform(inSpiral_->getLocalPoint(sStartArc), inSpiral_->getLocalHeading(sStartArc));
//	addTrackComponent(arc_);
//
//	delTrackComponent(outSpiral_);
//	double sEndArc = arc_->getSEnd();
//	outSpiral_->setSStart(sEndArc);
//	outSpiral_->setLength(L);
//	outSpiral_->setCurvStart(-Curv, false);
//	outSpiral_->setLocalTransform(arc_->getLocalPoint(sEndArc), arc_->getLocalHeading(sEndArc));
//	addTrackComponent(outSpiral_);

#endif

#if 0
	// SYMMETRIC CIRCULAR ARC INSERTION //
	//

	// Distance between inSpiral start and outSpiral end //
	//
	double DistAB = QVector2D(outSpiral_->getLocalPoint(outSpiral_->getSEnd()) - inSpiral_->getLocalPoint(inSpiral_->getSStart())).length();
//	qDebug() << "DistAB: " << DistAB;

	// Angle between the tangents (Oeffnungswinkel) //
	//

	double gamma = fabs(outSpiral_->getLocalHeadingRad(outSpiral_->getSEnd()) - inSpiral_->getLocalHeadingRad(inSpiral_->getSStart()));
//	qDebug() << "Gamma: " << gamma;

	//  //
	//
	double T = DistAB / (2.0*cos(gamma/2.0));


	double tauMax = gamma/2.0;

// Variabel:
	double tau = tauMax * factor;
	if(tau < 0.001) tau = 0.001;
//	double tau = gamma/4.0;
//	double tau = fabs(inSpiral_->getLocalHeadingRad(inSpiral_->getSEnd()) - inSpiral_->getLocalHeadingRad(inSpiral_->getSStart()));
//	double tau = tauMax;
//	double tau = 0.01;
//	qDebug() << "tau: " << tau;

	double Curv = (sqrt(2.0*tau)*(TrackElementSpiral::y(sqrt(2.0*tau))*tan(gamma/2.0) + TrackElementSpiral::x(sqrt(2.0*tau))) + cos(tau)*tan(gamma/2.0) - sin(tau)) / T;

//	double A = sqrt(2.0*tau)/Curv;
//	qDebug() << "A: " << A;

	double L = 2.0*tau / Curv;
//	qDebug() << "L: " << L;

	double LArc = 2.0 * (gamma/2.0 - tau) / Curv;

	Curv *= inSpiral_->getAsign();

	inSpiral_->setLength(L);
	inSpiral_->setCurvEnd(Curv, false);

	delTrackComponent(arc_);
	double sStartArc = inSpiral_->getSEnd();
	arc_->setSStart(sStartArc);
	arc_->setLength(LArc);
	arc_->setCurvature(Curv);
	arc_->setLocalTransform(inSpiral_->getLocalPoint(sStartArc), inSpiral_->getLocalHeading(sStartArc));
	addTrackComponent(arc_);

	delTrackComponent(outSpiral_);
	double sEndArc = arc_->getSEnd();
	outSpiral_->setSStart(sEndArc);
	outSpiral_->setLength(L);
	outSpiral_->setCurvStart(Curv, false);
	outSpiral_->setLocalTransform(arc_->getLocalPoint(sEndArc), arc_->getLocalHeading(sEndArc));
	addTrackComponent(outSpiral_);

#endif

#if 0

{


		double tauApprox = tauOld;
		double crit = 0.0001; // stop if improvement is less than ...
		int iIteration = 0;

		double tau1 = tauOld*0.9;
	qDebug() << "resultQ: " << q(tauOld, tau1) << " " << tauOld;
	qDebug() << "resultQ1: " << q(tau1, tau1) << " " << tau1;
	qDebug() << "resultQa1: " << q(angle_-tau1, tau1) << " " << tau1 << " " << angle_;
		do
		{
			tauOld = tauApprox;
			++iIteration;

			tauApprox = tauOld - q(tauOld, tau1)/dq(tauOld);

			qDebug() << "tauApprox: " << q(tauApprox, tau1) << " " << dq(tauApprox) << " " << tauApprox;

			// Check if it's ok to continue //
			//
			if(tauApprox < 0.0)
			{
				qDebug("kleiner null");
			}
//			else if(tauApprox > )
//			{
//				qDebug(" ");
//			}
			else if(iIteration >= 50)
			{
				qDebug("zu lang");
			}
		}
		while(fabs(tauOld - tauApprox) > crit);


	qDebug() << "result2Q: " << q(tauApprox, tau1) << " " << tauApprox << " in " << iIteration;

}

#endif
