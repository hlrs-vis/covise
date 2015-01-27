/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * shapes.h
 *
 *  Created on: Nov 10, 2012
 *      Author: asher
 */

#ifndef __OSGPCL_SHAPES_H__
#define __OSGPCL_SHAPES_H__

#include <osg/Geometry>

namespace osgpcl
{

/*
	 * Convience function for building a 1 meter long arrow geometry
	 * oriented along the z+ direction.
	 */
static bool buildArrowGeometry(osg::Geometry *geom);
}

#endif /* SHAPES_H_ */
