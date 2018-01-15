/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Construction.h"

#include <boost/foreach.hpp>

#include "Body.h"

using namespace std;

namespace KardanikXML
{

Construction::Construction()
{
}

const Construction::Bodies &
Construction::GetBodies() const
{
    return m_Bodies;
}
void
Construction::AddBody(std::shared_ptr<Body> body)
{
    m_Bodies.push_back(body);
    body->SetParentConstruction(shared_from_this());
}

const Construction::Joints &
Construction::GetJoints() const
{
    return m_Joints;
}

void
Construction::AddJoint(std::shared_ptr<Joint> joint)
{
    m_Joints.push_back(joint);
}

std::shared_ptr<Body>
Construction::GetBodyByName(const std::string &name) const
{
    BOOST_FOREACH (std::shared_ptr<Body> body, m_Bodies)
    {
        if (body && body->GetName() == name)
        {
            return body;
        }
    }
    return std::shared_ptr<Body>();
}

void Construction::SetNamespace(const std::string &theNamespace)
{
    m_TheNamespace = theNamespace;
}

const std::string &Construction::GetNamespace() const
{
    return m_TheNamespace;
}
}
