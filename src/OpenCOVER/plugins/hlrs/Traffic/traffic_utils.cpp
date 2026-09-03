#include "traffic_utils.h"

osg::Vec2 toVec2(osg::Vec3 v) { return osg::Vec2(v.x(), v.y()); }
float distanceRatio(osg::Vec2 x, osg::Vec2 y)
{
    auto l = y.length2();
    if (l == 0)
        return 0.0;
    return (x * y) / l;
}

/* utils for rotating angles the shortest way */
double angle_difference(double a, double b)
{
    return std::remainder(b - a, 2.0 * M_PI);
}
double lerp_angle(double a, double b, double f)
{
    return a + angle_difference(a, b) * f;
}

double unlerp(double a, double b, double f)
{
    return (f - a) / (b - a);
}
