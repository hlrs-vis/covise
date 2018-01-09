/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "BodyJointDesc.h"

using namespace std;

namespace KardanikXML
{

BodyJointDesc::BodyJointDesc()
{
}

BodyJointDesc::BodyJointDesc(std::shared_ptr<Body> body, std::shared_ptr<Point> point)
    : m_Body(body)
    , m_Point(point)
{
}

void BodyJointDesc::SetBody(std::shared_ptr<Body> body)
{
    m_Body = body;
}

std::shared_ptr<Body> BodyJointDesc::GetBody() const
{
    return m_Body;
}

void BodyJointDesc::SetPointInBody(std::shared_ptr<Point> point)
{
    m_Point = point;
}

std::shared_ptr<Point> BodyJointDesc::GetPoint() const
{
    return m_Point;
}
}
