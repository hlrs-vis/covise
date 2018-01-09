/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * BodyJointDesc.h
 *
 *  Created on: Jan 2, 2012
 *      Author: jw_te
 */

#pragma once

#include <memory>

namespace KardanikXML
{

class Body;
class Point;

class BodyJointDesc
{
private:
public:
    BodyJointDesc();
    BodyJointDesc(std::shared_ptr<Body> body, std::shared_ptr<Point> point);

    void SetBody(std::shared_ptr<Body> body);
    std::shared_ptr<Body> GetBody() const;

    void SetPointInBody(std::shared_ptr<Point> point);
    std::shared_ptr<Point> GetPoint() const;

private:
    std::shared_ptr<Body> m_Body;
    std::shared_ptr<Point> m_Point;
};
}
