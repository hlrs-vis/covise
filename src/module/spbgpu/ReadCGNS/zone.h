/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * zone.h
 *
 *	CGNS Zone reader
 *
 *  Created on: 25.01.2010
 *      Author: Vlad
 */

#ifndef ZONE_H_
#define ZONE_H_

#include <vector>
//#include <string>
#include <cgnslib.h>
#include <do/coDoUnstructuredGrid.h> //for grid element types
#include <do/coDoData.h>
#include "common.h"

#include <api/coModule.h> //only for Send*** output!
using namespace covise;

class zone
{
    int error{0}; /// Error return for CGNS lib

    //from COMODULE
    enum
    {
        FAIL = -1,
        SUCCESS = 0 //return values
    };

    //CGNS indices for zone

    int index_file;  /// CGNS file index
    int ibase;      /// CGNS base index
    int izone;      /// CGNS zone index
    params p;

    vector<float> fx; /// COVISE X coord array
    vector<float> fy; /// COVISE Y coord array
    vector<float> fz; /// COVISE Z coord array

    vector<int> conn; /// COVISE connectivity array
    vector<int> tl;   /// COVISE type list arraay
    vector<int> elem; /// COVISE element list array

    vector<float> scalar[4]; /// COVISE scalar fields

    //velocity vector

    vector<float> fvx;  /// COVISE velocity x
    vector<float> fvy;  /// COVISE velocity y
    vector<float> fvz;  /// COVISE velocity z

    cgsize_t zonesize[9]; /// CGNS 3 sizes for unstructured grid; max 9 for 3d structured

    char zonename[100];  /// CGNS zone name

    CGNS_ENUMT(ZoneType_t) zonetype; /// CGNS zone type

    //solution-related
    int read_field(int isol, int ifield, cgsize_t start, cgsize_t end, vector<float> &fvar);

    //coord related
    int read_coords();

    //section related
    int read_one_section(int isection);
    int select_sections(vector<int> &sections);
    static bool IsMixed3D(const vector<cgsize_t> &conntemp);

public:
    //zone ();
    zone(int i_file, int i_base, int i_zone, params _p);
    int read();

    coDoUnstructuredGrid *create_do_grid(const char *usgobjname);
    coDoVec3 *create_do_vec(const char *velobjname);
    coDoFloat *create_do_scalar(const char *floatobjname, int n);
    coDistributedObject *create_do(const char *name, int type, int scal_no = 0);
};

#endif /* ZONE_H_ */
