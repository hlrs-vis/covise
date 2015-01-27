/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/coExport.h>
#include <string>
#include <do/coDoData.h>
#include <do/coDoLines.h>

#ifndef YAC
#include "coColors.h"
#endif

#define S_U 1
#define S_V 2
#define S_DATA 3
#define STR_GRD 1
#define RCT_GRD 2
#define UNI_GRD 3
#define on_the_bottom 1
#define on_the_middle 2

namespace covise
{

class coDistributedObject;
class coDoLines;
class coDoColormap;
class ScalarContainer;

class ALGEXPORT coVectField
{
private:
    float *s_out;
    int num_scalar;
    coDoLines *lines_out;
    coDoFloat *u_scalar_data;

    //data for arrows
    float angle_;
    float cos_a_;
    float sin_a_;
    float *cosenos_;
    float *senos_;
    float arrow_factor_;
    int num_sectors_;
    int numc_line_2_;
    int numc_line_3_;

    int numc;
    int length_param, fasten_param;
    float scale, scale_min, scale_max;
    int i_dim, j_dim, k_dim;
    float min_max[6];
    int grdtype;
    float *x_in, *x_c;
    float *y_in, *y_c;
    float *z_in, *z_c;
    float *s_in;
    float *u_in, *v_in, *w_in;
    int *l_l, *v_l;
    float *n_x, *n_y, *n_z;

    void create_strgrid_lines();
    void create_rectgrid_lines();
    void create_unigrid_lines();
    void create_lines();
    void vector_displacement();

    void create_stings();
    void fillTheStingPoints(const int);
    void orthoBase(float *, float *, const float *);

    void project_lines(int keepLength);

public:
    /// Unstructured Grid C'tor
    coVectField(int num_points,
                float *_x_in, float *_y_in, float *_z_in,
                float *_u_in, float *_v_in, float *_w_in);

    /// Structured Grid C'tor
    coVectField(int grd_type,
                float *_x_in, float *_y_in, float *_z_in,
                float *_u_in, float *_v_in, float *_w_in,
                int _i_dim, int _j_dim, int _k_dim);

    /// Unigrid C'tor
    coVectField(float *_x_in, float *_y_in, float *_z_in,
                float *_u_in, float *_v_in, float *_w_in,
                int _i_dim, int _j_dim, int _k_dim,
                float min_max[6]);

    /// set the scalar field
    void setScalarInField(float *scField)
    {
        s_in = scField;
    }

    /// set normal list (to project lines)
    void setProjectionNormals(float *_n_x, float *_n_y, float *_n_z)
    {
        n_x = _n_x;
        n_y = _n_y;
        n_z = _n_z;
    }

    virtual ~coVectField()
    {
        if (s_out)
            delete[] s_out;
    }
    enum ValFlag
    {
        PER_VERTEX,
        PER_LINE
    };
    void compute_vectorfields(float scale, int length_param, int fasten_param, int num_sectors,
                              const coObjInfo *outlines, const coObjInfo *outfloat, ValFlag flag = PER_VERTEX);
    void compute_vectorfields(float scale_, int length, int fasten_param, int num_sectors,
                              float arrow_factor, float angle, const coObjInfo *objInfoLines, const coObjInfo *objInfoFloat, ValFlag = PER_VERTEX);

    float *get_scalar_data()
    {
        return s_out;
    }
    int get_scalar_count()
    {
        return num_scalar;
    }
    coDoLines *get_obj_lines()
    {
        return lines_out;
    }
    coDoFloat *get_obj_scalar()
    {
        return u_scalar_data;
    }
};

#ifndef YAC
class ALGEXPORT coDistrVectField
{
public:
    coDistrVectField(const coDistributedObject *geo, const coDistributedObject *vect,
                     const coDoColormap *colorMap, float scale, int lineChoice, int numsectors, int projectlines);
    coDistrVectField(const coDistributedObject *geo, const coDistributedObject *vect,
                     const coDoColormap *colorMap, float scale, int lineChoice, int numsectors, float arrow_factor,
                     float angle, int projectlines);

    bool Execute(coDistributedObject **lines,
                 coDistributedObject **colorSurf, coDistributedObject **colorLines,
                 std::string, std::string, bool, const ScalarContainer *, int vectOpt);

private:
    bool CreateLinesAndScalar(coDistributedObject **lines, std::string linesName,
                              const coDistributedObject *geo, const coDistributedObject *vect,
                              ScalarContainer &scalar);
    bool CreateColors(coDistributedObject **color, std::string colorName,
                      const ScalarContainer &scalar,
                      bool CMAPAttr, const ScalarContainer *SCont, int repeat);
    const coDistributedObject *_geo;
    const coDistributedObject *_vect;
    const coDoColormap *_colorMap;

    // parameters for coVectField
    float _scale;
    int _length_param;
    int _fasten_param;
    int _num_sectors;
    float _arrow_factor;
    float _angle;
    int _project_lines;
};
#endif
}
