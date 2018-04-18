/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * Body.h
 *
 *  Created on: Jan 2, 2012
 *      Author: jw_te
 */

#ifndef ANCHOR_H_
#define ANCHOR_H_

#include <string>
#include <vector>
#include <memory>
namespace KardanikXML
{

class Point;
class Body;

class Anchor
{
public:
    Anchor();

    std::string GetAnchorNodeName() const;
    void SetAnchorNodeName(std::string name);

    std::shared_ptr<Point> GetAnchorPoint() const;
    void SetAnchorPoint(std::shared_ptr<Point> point);

    std::weak_ptr<Body> GetParentBody() const;
    void SetParentBody(std::weak_ptr<Body> parent);

private:
    std::string m_AnchorNodeName;
    std::shared_ptr<Point> m_AnchorPoint;
    std::weak_ptr<Body> m_ParentBody;
};
}

#endif /* ANCHOR_H_ */
