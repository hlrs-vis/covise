/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * PointRelative.h
 *
 *  Created on: Aug 1, 2012
 *      Author: jw_te
 */

#ifndef POINTRELATIVE_H_
#define POINTRELATIVE_H_

#include <string>
#ifndef WIN32
#include <boost/tr1/memory.hpp>
#endif
#include <memory>
#include "Point.h"

namespace KardanikXML
{

class PointRelative : public Point
{
private:
public:
    PointRelative();
    PointRelative(const std::string &pointRefID, float xOffset, float yOffset, float zOffset);

    void SetXOffset(float x);
    float GetXOffset() const;

    void SetYOffset(float y);
    float GetYOffset() const;

    void SetZOffset(float z);
    float GetZOffset() const;

    void ResolveAgainst(std::tr1::shared_ptr<Point> referencePoint);
    const std::string &GetPointRefID() const;

private:
    std::string m_PointRefID;

    float m_XOffset;
    float m_YOffset;
    float m_ZOffset;
};
}

#endif /* POINTRELATIVE_H_ */
