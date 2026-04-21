/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVER_GRID_H
#define COVER_GRID_H

#include <osg/Geode>
#include <osg/Vec4>
#include <osg/ref_ptr>

#include <util/coExport.h>

namespace opencover
{
struct GridProps
{
    /// world units from center to edge
    float halfExtent = 10.5f;

    /// distance between grid lines
    float spacing = 1.0f;

    // Base color of the grid lines
    osg::Vec4 gridColor = osg::Vec4(0.6, 0.6, 0.6, 0.4);

    // Color of the primary axis lines, in order X-Y-Z (usually red-green-blue)
    osg::Vec4 axisColors[3] = {
        osg::Vec4(0.8, 0.2, 0.2, 1.0),
        osg::Vec4(0.2, 0.8, 0.2, 1.0),
        osg::Vec4(0.2, 0.2, 0.8, 1.0),
    };

    // Line width for normal grid lines
    float thinLineWidth = 1.0f;

    // Line width for axes
    float thickLineWidth = 4.f;

    bool renderOnTop = false; // if true, draws above other geometry
};

COVEREXPORT osg::ref_ptr<osg::Geode> createGrid(GridProps props = GridProps { });
}

#endif
