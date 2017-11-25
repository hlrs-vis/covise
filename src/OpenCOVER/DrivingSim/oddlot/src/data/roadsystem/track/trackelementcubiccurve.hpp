/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   10/21/2010
**
**************************************************************************/

#ifndef TRACKELEMENTCUBICCURVE_HPP
#define TRACKELEMENTCUBICCURVE_HPP

#include "trackelement.hpp"
#include "src/util/math/polynomial.hpp"

class TrackElementCubicCurve : public TrackElement
{

    //################//
    // STATIC         //
    //################//

public:
    enum TrackElementCubicCurveChange
    {
        CTCC_ParameterChange = 0x1,
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    explicit TrackElementCubicCurve(double x, double y, double angleDegrees, double s, double length, Polynomial *polynomialU, Polynomial *polynomialV, const QString &pRange);
    virtual ~TrackElementCubicCurve();

	Polynomial *getPolynomialU()
	{
		return polynomialU_;
	}

	Polynomial *getPolynomialV()
	{
		return polynomialV_;
	}

	QString getPRange()
	{
		return pRange_;
	}

	// Track Component //
	//
	virtual double getCurvature(double s);

	virtual QPointF getPoint(double s, double d = 0.0);
	virtual double getHeading(double s);
	virtual double getHeadingRad(double s);

	virtual QPointF getLocalPoint(double s, double d = 0.0);
	virtual double getLocalHeading(double s);
	virtual double getLocalHeadingRad(double s);
	virtual QVector2D getLocalTangent(double s);
	virtual QVector2D getLocalNormal(double s);

	virtual int getStartPosDOF() const
	{
		return 2;
	}
	virtual int getEndPosDOF() const
	{
		return 2;
	}
	virtual int getStartRotDOF() const
	{
		return 1;
	}
	virtual int getEndRotDOF() const
	{
		return 1;
	}

	virtual void setLocalStartPoint(const QPointF &startPoint);
	virtual void setLocalEndPoint(const QPointF &endPoint);
	virtual void setLocalStartHeading(double startHeading);
	virtual void setLocalEndHeading(double endHeading); 

    // Observer Pattern //
    //
    virtual void notificationDone();
    int getTrackElementCubicCurveChanges() const
    {
        return trackElementCubicCurveChanges_;
    }
    void addTrackElementCubicCurveChanges(int changes);

    // Prototype Pattern //
    //
    virtual TrackComponent *getClone() const;
	TrackElementCubicCurve *getClonedCubicCurve() const;

    // Visitor Pattern //
    //
    virtual void accept(Visitor *visitor);

protected:
private:
	TrackElementCubicCurve(); /* not allowed: use clone() instead */
	TrackElementCubicCurve(const TrackElementCubicCurve &); /* not allowed: use clone() instead */
	TrackElementCubicCurve &operator=(const TrackElementCubicCurve &); /* not allowed: use clone() instead */

	double f(double a, double b, double c, double d, double t) const; // value
	double df(double b, double c, double d, double t) const; // slope
	double ddf(double c, double d, double t) const; // curvature (2nd derivative)

	double getT(double s);
	double getParametricCurveLength(double from, double to);
	double parametricF(double x, double factor);
	double hdgRad(double x) const; // slope angle (heading) in [degrees]

	void getControlPoints(double a, double b, double c, double d, double &p0, double &p1, double &p2, double &p3);
	void setParametersFromControlPoints(double &a, double &b, double &c, double &d, double p0, double p1, double p2, double p3);

    //################//
    // PROPERTIES     //
    //################//

private:
    // Observer Pattern //
    //
    int trackElementCubicCurveChanges_;

	Polynomial *polynomialU_;
	Polynomial *polynomialV_;

	double Va_;
	double Vb_;
	double Vc_;
	double Vd_;
	double Ua_;
	double Ub_;
	double Uc_;
	double Ud_;
	QString pRange_;

};

#endif // TRACKELEMENTCUBICCURVE_HPP
