/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
//  ANSYS element data base
//
//  Initial version: 2006-5-12 Sven Kufer
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2006 by Visenso
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _ANSYS_H_
#define _ANSYS_H_

#include <do/coDoUnstructuredGrid.h>

class ANSYS
{
public:
    ANSYS();

    /** Get the object reference for
       *  accessing the internal data
       *
       *  @return   object reference
       */
    static ANSYS &get_handle();

    static const int LIB_SIZE = 1024;
    static const int TYPE_TARGET = -8327465;
    static const int TYPE_TARGET_2D = -8327466;

    // New datatypes for supporting the ANSYS Element Library
    static const int TYPE_4_NODE_PLANE = -4; // We use this value instead of 4 to distinguish from TYPE_TETRAHEDER = 4
    static const int TYPE_8_NODE_PLANE = 8; // For all the other cases the value represents the number of nodes
    static const int TYPE_10_NODE_SOLID = 10;
    static const int TYPE_20_NODE_SOLID = 20;

    enum StressSupport
    {
        SOLID,
        SHELL,
        AXI_SHELL,
        PLANE,
        BEAM3,
        BEAM4,
        LINK,
        THERMAL_SOLID,
        THERMAL_PLANE,
        NO_STRESS
    };
    enum SHELL_RESULT
    {
        TOP,
        BOTTOM,
        AVERAGE
    };

    int ElementType(int routine, int noCovNodes);

    StressSupport getStressSupport(int routine);
    int getCovType(int routine);

protected:
private:
    // singleton object
    static ANSYS *ansys_;

    int CovType_[LIB_SIZE];
    StressSupport StressSupport_[LIB_SIZE];
    int ANSYSNodes_[LIB_SIZE];
};
#endif
