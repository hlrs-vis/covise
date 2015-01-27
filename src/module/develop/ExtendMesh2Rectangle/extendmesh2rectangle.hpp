/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef EXTENDMESH2RECTANGLE_HPP_WEDDEC21114004CET2005
#define EXTENDMESH2RECTANGLE_HPP_WEDDEC21114004CET2005

/*

   MODULE ExtendMesh2Rectangle

   The input grid is expected to be located after z-axis-projection in
   the north-east-quarter of the x-y-coordinate-system.  Further the
   grid should be alligned with the x and y-axis.  (See the sketch
   below.)

   Sketch (input-grid):

   + y_ne
   |
   |
   |---------
   |	     \
   |   	      \
   |   	       \
   | grid      	|
   |  	 	|
   0------------------+
    		    x_ne


   Output is an extension of grid that fills the spcified rectangle
   with the north-east point x_ne, y_ne.

   Sketch (output-grid):

   y_ne
   +------------------+
   |  gridded	      |
   |	 somehow      |
   |---------	      |
   |	     \	      |
   |	      \	      |
   |	       \      |
   |  grid     	|     |
   |		|     |
   0------------------+

   
   There is an option to control the 'gridded somehow' region.  If
   parameter radius is > 0 the border of the grid gets projected on
   the circle with radius and the grid is extented to that circle.
   Then the meshing builds on this intermediate mesh.
*/

#include <api/coSimpleModule.h>

class ExtendMesh2Rectangle : public coSimpleModule
{
    coFloatSliderParam *para_xSize_;
    coFloatSliderParam *para_ySize_;
    coFloatSliderParam *para_radius_;

    coInPort *p_in_geometry_;
    coOutPort *p_out_geometry_;

    const float DEFAULT_MAXIMUM_;
    float xSize_;
    float ySize_;
    float radius_;

    virtual int compute();

    bool featureRadiusEnabled() const
    {
        return radius_ >= 0;
    }

public:
    ExtendMesh2Rectangle();
};

// local variables:
// mode: c++
// end:

#endif
