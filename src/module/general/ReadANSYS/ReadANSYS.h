/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  MODULE ReadANSYS
//
//  ANSYS reader
//
//  Initial version: 2002-2-01 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2001 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _READ_ANSYS_H_
#define _READ_ANSYS_H_

// #define _TEST_RST_FILE_NAME_

#include <api/coModule.h>
using namespace covise;
#include <vector>
#include <api/coHideParam.h>
#include "ReadRST.h"
#include "Map1D.h"
#include "DOFOptions.h"

class ReadANSYS : public coModule
{
public:
    ReadANSYS(int argc, char *argv[]);
    virtual void param(const char *paramName, bool inMapLoading);
    static const int V_OFFSET = 500;
    static const int EX_OFFSET = 1000;

protected:
private:
    char *oldFileName;
    int getNumberOfNodes(int elem, int routine);
    enum FieldType
    {
        SCALAR,
        VECTOR
    };
    enum SHELL_RESULT
    {
        TOP,
        BOTTOM,
        AVERAGE
    };

    // make covise objects for DOF data
    void MakeGridAndObjects(const std::string &gridName,
                            std::vector<int> &el,
                            std::vector<int> &vl,
                            std::vector<float> &x_l,
                            std::vector<float> &y_l,
                            std::vector<float> &z_l,
                            std::vector<int> &t_l,
                            const std::string &dataName,
                            const float *const *field,
                            FieldType ftype,
                            const std::string &matName,
                            const int *materials,
                            std::vector<coDistributedObject *> &grid_set_list,
                            std::vector<coDistributedObject *> &data_set_list,
                            std::vector<coDistributedObject *> &mat_set_list);
    // make covise objects for derived data
    void MakeGridAndObjectsElems(const std::string &gridName,
                                 std::vector<int> &el,
                                 std::vector<int> &vl,
                                 std::vector<float> &x_l,
                                 std::vector<float> &y_l,
                                 std::vector<float> &z_l,
                                 std::vector<int> &t_l,
                                 const std::string &dataName,
                                 const float *field0,
                                 const float *field1,
                                 const float *field2,
                                 const std::string &matName,
                                 const int *materials,
                                 const std::string &nodeIndName,
                                 std::vector<coDistributedObject *> &grid_set_list,
                                 std::vector<coDistributedObject *> &data_set_list,
                                 std::vector<coDistributedObject *> &mat_set_list,
                                 std::vector<coDistributedObject *> &node_decode_set_list,
                                 bool data_per_elem);

    // Returns 1 when retrieving derived, vector data
    int outputIsVector();

    DOFOptions DOFOptions_;
    // Reads choices for DOF data available in the file
    int SetNodeChoices();

    int file_ok_;

    int open_err_;
    bool inMapLoading;
    ReadRST readRST_;
    virtual int compute(const char *port);
    virtual void postInst();

    int extractName(std::string &);
    int fileNameChanged(int yell);
    void outputDummies();
    std::vector<coUifPara *> allFeedbackParams_;

    int onlyGeometry();
    int nodalData();
    int derivedData();

    // ElemValues[ANSYSelemCode][0] : all scalar values for elem ANSYSelemCode
    // ( ElemValues[ANSYSelemCode][0],
    //    ElemValues[ANSYSelemCode][1],
    //    ElemValues[ANSYSelemCode][2] ) : all vector values for elem ANSYSelemCode
    typedef vector<vector<float> > ElemValues;

    // NodeValues[ANSYScode][0] : all scalar values for node ANSYScode
    // vector values similiar to ElemValues
    typedef map<int, vector<vector<float> > > NodeValues;

    // AvgNodeValues[cov_index][0]: averaged values for COVISE node cov_index
    // vector values similiar to ElemValues
    typedef map<int, vector<float> > AvgNodeValues;

    vector<float> ProcessDerivedField(std::vector<double> &derData, int routine);

    std::vector<float> displacements_[3];
    void ReadDisplacements(const Map1D &nodeDecode);
    void AddDisplacements(std::vector<float> &x_l, std::vector<float> &y_l, std::vector<float> &z_l);

    void workOutEqv(std::vector<double> &, std::vector<double> &);

    int eqvStrainNotWritten; // flag if "the" bug of ANSYS appears
    // Parameters
    coFileBrowserParam *p_rst_;
    coIntVectorParam *p_times_;
    coFloatParam *p_scale_;
    coChoiceParam *p_sol_;
    coChoiceParam *p_nsol_;
    coChoiceParam *p_esol_;
    coChoiceParam *p_stress_;
    coChoiceParam *p_beam_stress_;
    coChoiceParam *p_axi_shell_stress_;
    coChoiceParam *p_top_bottom_;
    coChoiceParam *p_thermalFlux_;
    coChoiceParam *p_vol_energy_;
    coChoiceParam *p_mag_flux_dens_;
    coBooleanParam *p_output_node_decode_;
    coBooleanParam *p_vertex_based_;

    // hide all params but p_rst_
    coHideParam *h_times_;
    coHideParam *h_scale_;
    coHideParam *h_sol_;
    coHideParam *h_nsol_;
    coHideParam *h_esol_;
    coHideParam *h_stress_;
    coHideParam *h_beam_stress_;
    coHideParam *h_axi_shell_stress_;
    coHideParam *h_top_bottom_;
    coHideParam *h_thermalFlux_;
    coHideParam *h_vol_energy_;
    coHideParam *h_mag_flux_dens_;
    coHideParam *h_output_node_decode_;
    coHideParam *h_vertex_based_;
    std::vector<coHideParam *> hparams_;
    void useReadANSYSAttribute(const coDistributedObject *);

    // Ports
    std::string FileName_;
    coInputPort *p_file_name_;

    coOutputPort *p_grid_;
    coOutputPort *p_field_;
    coOutputPort *p_materials_;
#ifdef _TEST_RST_FILE_NAME_
    coOutputPort *p_outname_;
#endif
    //   coOutputPort *p_node_decode_;
};
#endif
