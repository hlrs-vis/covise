/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COVER_GEODATA_GEODATA_H
#define _COVER_GEODATA_GEODATA_H

#include <string>
#include <string_view>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <osg/Matrix>

#include <util/coExport.h>

#ifdef HAVE_PROJ
#include <proj.h>
#endif

#define GEODATA_DEFAULT_PROJECTION_GLOBAL "EPSG:4326"
#define GEODATA_DEFAULT_PROJECTION_LOCAL "EPSG:25832"

namespace opencover
{

class COVRGEODATAEXPORT GeoData
{
public:
    GeoData();

    void setProjection(std::string_view projection);
    void setProjectOffset(const glm::vec3 &projectOffset);

    const std::string &projection() const;
    const glm::vec3 &projectOffset() const;

    /// Transform a global position (easting-northing-altitude) into local
    /// (x-y-z) coordinates, applying transform and project offset.
    glm::vec3 toLocal(const glm::vec3 &global_position) const;

    /// Transform a local (x-y-z) coordinate into a global position
    /// (easting-northing-altitude), applying the reverse project offset and
    /// transformation.
    glm::vec3 toGlobal(const glm::vec3 &local_position) const;

protected:
#ifdef HAVE_PROJ
    PJ *m_coordTransformation;
#endif

    std::string m_projection;
    glm::vec3 m_projectOffset;
};

}
#endif
