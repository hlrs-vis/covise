/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Anchor.h"

#include <stdexcept>
#include <string>

#include <boost/foreach.hpp>

#include "Point.h"
#include "Body.h"

using namespace std;
using namespace std::tr1;

namespace KardanikXML
{

Anchor::Anchor()
{
}

std::tr1::shared_ptr<Point> Anchor::GetAnchorPoint() const
{
    return m_AnchorPoint;
}

void Anchor::SetAnchorPoint(std::tr1::shared_ptr<Point> point)
{
    m_AnchorPoint = point;
}

string Anchor::GetAnchorNodeName() const
{
    return m_AnchorNodeName;
}

void Anchor::SetAnchorNodeName(string nodeName)
{
    m_AnchorNodeName = nodeName;
}

std::tr1::weak_ptr<Body> Anchor::GetParentBody() const
{
    return m_ParentBody;
}

void Anchor::SetParentBody(std::tr1::weak_ptr<Body> parent)
{
    m_ParentBody = parent;
}
}
