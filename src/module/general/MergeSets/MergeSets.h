/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _MERGEPOLYGONS_NEW_H
#define _MERGEPOLYGONS_NEW_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                (C)2013 Stellba Hydro GmbH & Co. KG  ++
// ++ Description:  MergeSets module                                  ++
// ++                                                                     ++
// ++ Author:                                                             ++
// ++                                                                     ++
// ++                            Martin Becker                            ++
// ++                     Stellba Hydro GmbH & Co. KG                     ++
// ++                            Eiffelstr. 4                             ++
// ++                        89542 Herbrechtingen                         ++
// ++                                                                     ++
// ++ Date:  09/2013  V1.0                                                ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <api/coSimpleModule.h>
#include <util/coviseCompat.h>
#include <float.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>

using namespace covise;

class MergeSets : public coSimpleModule
{

private:
    virtual int compute(const char *port);

    // parameters

    // ports
    coInputPort *p_inGeo;
    coInputPort *p_inScalarData;
    coInputPort *p_inVectorData;
    coOutputPort *p_outGeo;
    coOutputPort *p_outScalarData;
    coOutputPort *p_outVectorData;

    //private data
    bool havePolygons;
    bool hasTypeList;
    bool haveScalarData;
    bool haveVectorData;

    std::vector<int> container_conn_list;
    std::vector<int> container_element_list;
    std::vector<int> container_type_list;
    std::vector<float> container_x;
    std::vector<float> container_y;
    std::vector<float> container_z;
    std::vector<float> container_scalarData;
    std::vector<float> container_vectorData_X;
    std::vector<float> container_vectorData_Y;
    std::vector<float> container_vectorData_Z;

    int num_points_handled;
    int num_corners_handled;
    int num_elements_handled;

    int num_scalarData_handled;
    int num_vectorData_handled;

    int num_sets_handled;

protected:
    virtual void preHandleObjects(coInputPort **);
    virtual void postHandleObjects(coOutputPort **);

public:
    MergeSets(int argc, char *argv[]);
    virtual ~MergeSets();
};
#endif