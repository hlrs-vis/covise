#ifndef _TRAFFIC_UTILS_H
#define _TRAFFIC_UTILS_H

#include <osg/Vec2>
#include <osg/Vec3>

template <typename T>
T cubic_bezier(T p0, T p1, T p2, T p3, float t)
{
    float t1 = 1.f - t;
    return p0 * (t1 * t1 * t1) + p1 * (3 * t1 * t1 * t) + p2 * (3 * t1 * t * t) + p3 * (t * t * t);
}

template <typename T>
T cubic_bezier_tangent(T p0, T p1, T p2, T p3, float t)
{
    float t1 = 1.f - t;
    return p0 * (-3 * t1 * t1) + p1 * (3 * t1 * t1) - p1 * (6 * t * t1) - p2 * (3 * t * t) + p2 * (6 * t * t1) + p3 * (3 * t * t);
}

osg::Vec2 toVec2(osg::Vec3 v) { return osg::Vec2(v.x(), v.y()); }
float distanceRatio(osg::Vec2 x, osg::Vec2 y)
{
    return (x * y) / y.length2();
}

/* utils for rotating angles the shortest way */
double angle_difference(double a, double b)
{
    double difference = fmod(b - a, M_PI_2);
    return fmod(2.0 * difference, M_PI_2) - difference;
}
double lerp_angle(double a, double b, double f)
{
    return a + angle_difference(a, b) * f;
}

#endif
