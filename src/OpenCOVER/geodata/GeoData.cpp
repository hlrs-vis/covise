/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "GeoData.h"

using namespace opencover;

GeoData::GeoData()
{
    setProjection(GEODATA_DEFAULT_PROJECTION_LOCAL);
}

void GeoData::setProjection(std::string_view projection)
{
    m_projection = projection;

#ifdef HAVE_PROJ
    m_coordTransformation = proj_create_crs_to_crs(PJ_DEFAULT_CTX,
        GEODATA_DEFAULT_PROJECTION_GLOBAL,
        m_projection.c_str(),
        NULL);
#else
    std::cerr << "coVRGeoData: no projection support, geo referencing will not work" << std::endl;
#endif
}

void GeoData::setProjectOffset(const glm::vec3 &projectOffset)
{
    m_projectOffset = projectOffset;
}

const std::string &GeoData::projection() const
{
    return m_projection;
}
const glm::vec3 &GeoData::projectOffset() const
{
    return m_projectOffset;
}

glm::vec3 GeoData::toLocal(const glm::vec3 &global_position) const
{
#ifndef HAVE_PROJ
    return global_position;
#else
    PJ_COORD c;
    c.lpz.lam = global_position.x;
    c.lpz.phi = global_position.y;
    c.lpz.z = global_position.z;

    PJ_COORD c_out = proj_trans(m_coordTransformation, PJ_FWD, c);

    glm::vec3 position_transformed(c_out.enu.e, c_out.enu.n, c_out.enu.u);

    return position_transformed - m_projectOffset;
#endif
}

glm::vec3 GeoData::toGlobal(const glm::vec3 &local_position) const
{
#ifndef HAVE_PROJ
    return local_position;
#else
    auto p = local_position + m_projectOffset;

    PJ_COORD c;
    c.enu.e = p.x;
    c.enu.n = p.y;
    c.enu.u = p.z;

    PJ_COORD c_out = proj_trans(m_coordTransformation, PJ_INV, c);

    return glm::vec3(c_out.lpz.phi, c_out.lpz.lam, c_out.lpz.z);
#endif
}
