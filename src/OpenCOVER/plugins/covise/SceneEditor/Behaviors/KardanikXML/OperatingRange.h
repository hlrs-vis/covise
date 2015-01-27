/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * OperatingRange.h
 *
 *  Created on: Jul 5, 2012
 *      Author: jw_te
 */

#ifndef OPERATINGRANGE_H_
#define OPERATINGRANGE_H_

#include <string>
#include <vector>
#ifndef WIN32
#include <boost/tr1/memory.hpp>
#endif
#include <memory>
#include <osg/Vec4f>

namespace KardanikXML
{

class Point;
class Body;

class OperatingRange
{
public:
    OperatingRange();

    enum Axis
    {
        X_AXIS,
        Y_AXIS,
        Z_AXIS
    };

    void SetAxis(Axis axis);
    void SetAxis(const std::string &axis);
    Axis GetAxis() const;

    float GetRadius() const;
    void SetRadius(float radius);

    std::tr1::shared_ptr<Point> GetCenterPoint() const;
    void SetCenterPoint(std::tr1::shared_ptr<Point> point);

    void SetColor(const osg::Vec4f &color);
    void SetColor(const std::string &color);
    const osg::Vec4f &GetColor() const;

private:
    Axis m_Axis;
    float m_Radius;
    std::tr1::shared_ptr<Point> m_CenterPoint;
    osg::Vec4f m_Color;
};
}
#endif /* OPERATINGRANGE_H_ */
