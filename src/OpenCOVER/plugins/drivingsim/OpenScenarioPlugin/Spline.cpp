#include "Spline.h"
using namespace std;


Spline::Spline():
    n(10),
    splineTraj(n)
{
}

void Spline::poly3Spline(Position* initPos)
{

    x0 = initPos->referencePosition[0];
    y0 = initPos->referencePosition[1];

    x1 = initPos->absPosition[0];
    y1 = initPos->absPosition[1];

    a0 = -2*(y1-y0)*1/((x1-x0)*(x1-x0)*(x1-x0));
    b0 = 3*(y1-y0)*1/((x1-x0)*(x1-x0));

    step = (x1-x0)/n;

    for(int i = 0; i<n; ++i)
    {
        double xx = x0+(i+1)*step;
        double yy = a0*pow(xx-x0,3)+b0*pow(xx-x0,2)+y0;
        splineTraj[i] = osg::Vec3(xx,yy,0.0);
    }
}

osg::Vec3 Spline::getSplinePos(int i)
{
    return splineTraj[i];
}

