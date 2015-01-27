/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TASC_FLOW_TDI_H
#define _TASC_FLOW_TDI_H

/**************************************************************************\ 
 **                                                                        **
 **                                                                        **
 ** Description:  COVISE TascFlowTDI  application module                   **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  Sasha Cioringa, Thilo Krueger                                 **
 **                                                                        **
 **                                                                        **
 ** Date:  16.11.00                                                        **
\**************************************************************************/

//#include "coModule.h"
#include "FieldFile.h"
#include "BcfFile.h"

#define MAX_REGIONS 300
#define MAX_REGION_LENGHT 20
#define MAX_GRIDNAME_LENGHT 20

class TascFlow
{

private:
    // private data
    int *IWK; // TDI array for integers
    float *RWK; // TDI array for floats
    char *CWK;
    char **All_Regions;
    int nr, ni, nc; // TDI array for characters
    int tascflowError; // tascflow error
    char init_path[100];

    int reg, vector1, vector2, data1, data2, data3;
    int has_grd_file, has_rso_file, has_bcf_file, has_step_rso_file;
    char *gridpath, *rsopath, *step_rsopath, *bcfpath;
    const char *const *VectChoiceVal, *const *ScalChoiceVal, *const *VectFieldList;

    int nb_field_list, nb_regions;
    int nb_scal, nb_vect;
    int ngrids;
    int param_counter;

    Fields *main_field;
    BcfFile *bcf;

    void read_fields(char ***, int *);
    void read_regions(char ***, int *);
    int open_database(char *, char *, char *, int);

public:
    int compute();
    TascFlow(char *grd, char *rso, char *bcf, char *step);
    ~TascFlow();
};
#endif
