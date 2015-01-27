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

#pragma once

#include <string>
#include <vector>
#ifndef WIN32
#include <boost/tr1/memory.hpp>
#endif
#include <memory>

namespace KardanikXML
{

class Body;
class Joint;

class Construction : public std::tr1::enable_shared_from_this<Construction>
{
private:
    typedef std::vector<std::tr1::shared_ptr<Body> > Bodies;
    typedef std::vector<std::tr1::shared_ptr<Joint> > Joints;

public:
    Construction();

    const Bodies &GetBodies() const;
    void AddBody(std::tr1::shared_ptr<Body> body);

    const Joints &GetJoints() const;
    void AddJoint(std::tr1::shared_ptr<Joint> joint);

    std::tr1::shared_ptr<Body> GetBodyByName(const std::string &name) const;

    void SetNamespace(const std::string &theNamespace);
    const std::string &GetNamespace() const;

private:
    Bodies m_Bodies;
    Joints m_Joints;

    std::string m_TheNamespace;
};
}
