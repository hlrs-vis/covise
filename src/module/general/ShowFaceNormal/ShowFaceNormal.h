/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NORMAL_H
#define _NORMAL_H
/****************************************************************************\ 
 **                                                            (C)2000 RUS   **
 **                                                                          **
 ** Description: Calculates the normals of an Vector and a suitable point    **
 **                                                                          **
 ** Name:        NORMAL                                                      **
 ** Category:    TOOLS                                                       **
 **                                                                          **
 ** Author: A. Heinchen		                                            **
 **                                                                          **
 ** History:  								    **
 ** Dec-00       					       		    **
 **   --started writting this module                                         **
 **                                                                          **
\****************************************************************************/

#include <do/coDoPolygons.h>
#include <do/coDoPoints.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoData.h>
#include <api/coSimpleModule.h>
using namespace covise;

class ShowFaceNormal : public coSimpleModule
{

private:
    //member functions
    //compute callback
    virtual int compute(const char *port);

    //member data
    coDoPoints *CentralPoints(char *objectname, int numPoints);
    coDoVec3 *vectors(char *objectname, int num_values);
    coDoPolygons *polygon(const char *name, int num_points, int num_corners, int num_polygons);
    coDoTriangleStrips *strips(const char *name, int num_points, int num_corners, int num_strips);

    //Ports
    coInputPort *port_inPort;
    coOutputPort *normal_outport;
    coOutputPort *point_outport;

    //parameters

public:
    ShowFaceNormal(int argc, char *argv[]); // Constructor: module set-up
};
#endif
