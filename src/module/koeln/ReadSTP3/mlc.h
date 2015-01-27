/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MLC_H
#define MLC_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cmath>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "stp_const.h"
#include "data_struct.h"

#define MAXVOIS 20
#define MAX_LAMELLAS 40
#define MAX_PIXELS_X 90
#define PI 3.1415926535898
#define Lamella_Thickness 1.622
#define Pixel_efect 0.5
#define SID 1000.0

struct xy
{
    double x;
    double y;
};

struct xyz
{
    double x;
    double y;
    double z;
};

struct xyze
{
    double x;
    double y;
    double z;
    double err;
};

struct mat44_t
{
    double mat[4][4];
};

struct header_plan_t
{
    float RTP_File_Version;
    char Created_by[20];
    char Created_on[30];
    int Plan_Number;
    char Name_Patient[30];
    char Patient_Id[5];
    char Diagnose[37];
    char Diagnose_Key[37];
    char Name_of_Therapist[20];
    char Comment_Comment[250];
    int Number_Comment_Comment;
    char End_of_Comment[14];
    char Patient_Coordinate_System[40];
    char Device_Coordinate_System[20];
    char Dose_Calculate_By[30];
    char Dose_Calculate_On[30];
    char Dose_File[20];
    char Dose_Calculate_Base_On[37];
    int Number_of_Misc_Plan_Data;
    int Number_Of_Modifications;
    char Modification001_By[20];
    char Modification001_On[30];
    char Modification001_Old_Saved[30];
    char Kind_Field_weighting[10];
    char Use_MLC_Date_Dose_Calc[5];
    char Dose_Data[10];
    float Prescribed_Dose;
    float Prescribed_Dose_Target_Surf;
    float Reference_Point_xyz[3];
    float Max_Dose_xyz[3];
    float Normalisation_Dose;
    int No_Fractions;
    int Number_Fields;
};

struct header_field_t
{
    char Dummy_Collimator_f[36];
    char Dummy_Area[20];
    char Field_Name[20];
    int Field_Number;
    char Field_Type[5];
    float Field_Weight;
    char Irradation_Device[10];
    char Radation_Type[10];
    float Iso_Distance;
    char Multi_Leaf_Name[10];
    float MLC_Rotation;
    float MLC_Distance;
    float MLC_Thickness;
    char MLC_Variable_Leaf_Width[5];
    int MLC_No_Leaf_Pairs;
    float Leaf_Width;
    int No_Misc_Field_Data;
    char Beamgroup_Type[5];
    char Kind_Beam_Weighting[15];
    char Indiv_Parameter_Flags[20];
    float Relative_Beam_Weigth;
    char Dose_Data[10];
    float Monitor_Units;
    float Radiological_Depth;
    float Output_Factor;
    float Calib_tissue_Mass_Ratio;
    float Energy;
    float Target_Point[3];
    float Stereo_Target_Point[3];
    float Gantry_Angle_f;
    float Patient_Support_Angle_f;
    float Beam_Limiting_Device_Angle_f;
    char Beam_Limiting_Device_Type_f[5];
    float Field_Width_FX_f;
    float Field_Height_FY_f;
    float Wedge_Angle_f;
    char Wedge_Orientation_f[5];
    int Irregular_Field_Number_f;
    char Irregular_Field_File_f[15];
    int Number_MLC_Leaf_Pairs_f;
    float Leaf_Position[40][2];
    int Nr_Sub_Groups;
    int Sub_Group_Flag;
};

struct beam_t
{
    char Dummy_Collimator[29];
    char Dummy_Area[20];
    int Sub_Group_Number;
    char Beamgroup_Type[5];
    char Indivi_Paramet_Flags[11];
    float Relative_Beam_Weight;
    char Dose_Data[10];
    float Monitor_Units;
    float Radiological_Depth;
    float Output_Factor;
    float Calib_tissue_Mass_Ratio;
    float Gantry_Angle;
    float Patient_Support_Angle;
    float Collimator_Angle;
    char Beam_Limiting_Device_Type[5];
    float Field_Width;
    float Field_Height;
    float Wedge_Angle;
    char Wedge_Orientation[5];
    int Irregular_Field_Number;
    char Irregular_Field_File[15];
    int Number_MLC_Leaf_Pairs;
    float Leaf_Pair[40][2];
};

struct field_t
{
    header_field_t field_header;
    std::vector<beam_t> all_beam; // matrix of structure, initialy empty
};

struct comment_string
{
    char comment_comment[250];
};

struct plan_t
{
    char Complete_File[200];
    int is_loaded;
    header_plan_t plan_header;
    std::vector<field_t> all_field;
    comment_string my_comment;
    std::vector<comment_string> comment;
};

struct voi_t
{
    int voi_no;
    char voi_name[40];
    int voi_property;
    int voi_first_slice;
    int voi_last_slice;
    int voi_color;
    int number_points;
    std::vector<xyze> voi_points;
};

struct patient_struct
{
    char Patient_Directory[200];
    char Last_Directory[200];
    char Image_File[200];
    char Tra_File[200];
    char VOI_File[200];
    char Patient_Name[80];
    char Comment_image_header[80];
    int Resolution;
    int No_Slices;
    double Slice_Distance;
    double Pixel_size;
    double Z_Table[200];
    double Global_Tra_Matrix[4][4];
    double Rev_Global_Tra_Matrix[4][4];
    int Is_Patient_Loaded;
    int Is_Cube_Loaded;
    int Is_VOI_Loaded;
    int Target_VOI;
    voi_t all_voi[MAXVOIS];
    plan_t plan_1, plan_2; // debe de ser vector de plan
};

// estan declaradas(las 3 estructuras) en main program,no se pueden declarar aqui como globales da problemas(ya que al
// ser visibles para todos los modulos cuando se llama hay confusion!), pero
// necesitamos definir aqui my_field ya que lo utilizamos tambien en readplan.cpp,luego utilizamos
// extern para poder utilizarla pero no volver a declararla,(esto solo se puede hacer en .h),se puede escribir
// como extern o se puede definir como local variable en read.cpp, la cuestion es que al ser declarada como local en cpp
// y estar declarada en main ambas declaraciones no se refieren a la misma variable, al igual que pasa con dummy_line,
// no se exactamente que implicaciones tiene eso!!!

#include <qstring.h>

int read_ct_tran(struct patient_struct *ppatient, QString *Result);
int read_img_header(struct patient_struct *ppatient, QString *Result);
int import_voi(struct patient_struct *ppatient, QString *Result);
int read_plan(struct patient_struct *ppatient, QString *Result);
int Calculate_Output_Factor_XY(struct patient_struct *ppatient, QString *Result);
int modification(struct patient_struct *ppatient);
int write_plan(struct patient_struct *ppatient);
int reset_plan(struct plan_t *pplan, QString *Result);
int calculate_profile(struct patient_struct *ppatient, QString *Result);

#endif /* MLC_H */
