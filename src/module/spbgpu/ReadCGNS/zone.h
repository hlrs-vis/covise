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
#include <string>
#include <cgnslib.h>
#include <do/coDoUnstructuredGrid.h> //for grid element types
#include <do/coDoData.h>
#include "common.h"

#include <api/coModule.h> //only for Send*** output!
using namespace covise;

class zone
{
    //from COMODULE

    int error;
    enum
    {
        FAIL = -1,
        SUCCESS = 0 //return values
    };

    //vars
    int index_file;
    int ibase;
    int izone;
    params p;

    //coords array
    vector<float> fx;
    vector<float> fy;
    vector<float> fz;

    vector<int> conn; //connectivity array
    vector<int> tl; //type list arraay
    vector<int> elem; //element list array

    //4 float fields
    vector<float> scalar[4];

    //velocity vector
    vector<float> fvx;
    vector<float> fvy;
    vector<float> fvz;

    cgsize_t zonesize[3]; //3 sizes for unstructured grid

    char zonename[100];

    CGNS_ENUMT(ZoneType_t) zonetype;

    //solution-related
    int read_field(int isol, int ifield, cgsize_t start, cgsize_t end, vector<float> &fvar);

    //coord related
    int read_coords(vector<float> &fx, vector<float> &fy, vector<float> &fz);

    //section related
    int read_one_section(int isection, vector<int> &conn, vector<int> &elem, vector<int> &tl);
    int select_sections(vector<int> &sections);
    bool IsMixed3D(const vector<cgsize_t> &conntemp);

public:
    //zone ();
    zone(int i_file, int i_base, int i_zone, params _p);
    //int read (int index_file,int ibase,int izone);
    int read();

    coDoUnstructuredGrid *create_do_grid(const char *usgobjname);
    coDoVec3 *create_do_vec(const char *velobjname);
    coDoFloat *create_do_scalar(const char *floatobjname, int n);
    coDistributedObject *create_do(const char *name, int type, int scal_no = 0);
};

#endif /* ZONE_H_ */
