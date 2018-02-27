#include "Spline.h"
using namespace std;


Spline::Spline(Position* initPos)
{
    xx.reserve(10);
    yy.reserve(10);
    x0 = initPos->referencePosition[0];
    y0 = initPos->referencePosition[1];

    x1 = initPos->absPosition[0];
    y1 = initPos->absPosition[1];

    a0 = -2*(y1-y0)*1/((x1-x0)*(x1-x0)*(x1-x0));
    b0 = 3*(y1-y0)*1/((x1-x0)*(x1-x0));

    step = (x1-x0)/10;

    for(int i = 0; i<10; ++i)
    {
        xx[i] = x0+(i+1)*step;
    }

    for(int i = 0; i<10; ++i)
    {
        yy[i] = a0*pow(xx[i]-x0,3)+b0*pow(xx[i]-x0,2)+y0;
    }
}
