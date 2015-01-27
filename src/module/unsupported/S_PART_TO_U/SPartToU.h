/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE SPartToU
//
//  Transforms part of a Structured grid into an unstructured one
//
//  Initial version: 2002-4-26 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _S_PART_TO_U_
#define _S_PART_TO_U_

#define _DO_NOT_COMPRESS_IA_
#include <util/coIA.h>

#include <api/coSimpleModule.h>
using namespace covise;

class SPartToU : public coSimpleModule
{
public:
    SPartToU();

protected:
private:
    int size_x_;
    int size_y_;
    int size_z_;

    // never, never, never delete these pointers!!!!
    int *index_; // from coDoIntArr
    float *x_c_; // from struct. grid
    float *y_c_;
    float *z_c_;

    // Ports and params
    static const int NUM_DATA_PORTS = 3;
    coInputPort *p_inSGrid_;
    coInputPort *p_codes_;
    coInputPort *p_inData_[NUM_DATA_PORTS];
    coOutputPort *p_outUGrid_;
    coOutputPort *p_outData_[NUM_DATA_PORTS];
    coOutputPort *p_outPoly_;
    coOutputPort *p_outDataPoly_[NUM_DATA_PORTS];
    virtual int compute();
    int Diagnose();
    int IndicesToNode(int, int, int);
    void RegisterElements(int c_x, int c_y, int c_z, int *markElems);
    void RegisterNodes(int c_x, int c_y, int c_z, int *markNodes);

    virtual void copyAttributesToOutObj(coInputPort **input_ports,
                                        coOutputPort **output_ports, int i);
    virtual void createGridData(const ia<int> &);
};
#endif
