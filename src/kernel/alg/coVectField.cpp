/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVectField.h"
#include <sysdep/math.h>
#include <do/coDoLines.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoSet.h>

#ifndef YAC
#include "coColors.h"
#endif

using namespace covise;

coVectField::coVectField(int num_points, float *_x_in, float *_y_in, float *_z_in, float *_u_in, float *_v_in, float *_w_in)
    : angle_((float)(9.5 * M_PI / 180.0))
    , cos_a_((float)cos(angle_))
    , sin_a_((float)sin(angle_))
    , cosenos_(0)
    , senos_(0)
    , arrow_factor_(0.20f)
    , numc(num_points)
    , x_in(_x_in)
    , y_in(_y_in)
    , z_in(_z_in)
    , u_in(_u_in)
    , v_in(_v_in)
    , w_in(_w_in)
{
    i_dim = j_dim = k_dim = 0;
    s_out = NULL;
    num_scalar = 0;
    lines_out = NULL;
    u_scalar_data = NULL;
    s_in = NULL;
    n_x = n_y = n_z = NULL;
}

coVectField::coVectField(int grd_type, float *_x_in, float *_y_in, float *_z_in, float *_u_in, float *_v_in, float *_w_in, int _i_dim, int _j_dim, int _k_dim)
    : angle_((float)(9.5 * M_PI / 180.0))
    , cos_a_((float)cos(angle_))
    , sin_a_((float)sin(angle_))
    , cosenos_(0)
    , senos_(0)
    , arrow_factor_(0.20f)
    , i_dim(_i_dim)
    , j_dim(_j_dim)
    , k_dim(_k_dim)
    , grdtype(grd_type)
    , x_in(_x_in)
    , y_in(_y_in)
    , z_in(_z_in)
    , u_in(_u_in)
    , v_in(_v_in)
    , w_in(_w_in)

{
    numc = i_dim * j_dim * k_dim;
    s_out = NULL;
    num_scalar = 0;
    lines_out = NULL;
    u_scalar_data = NULL;
    s_in = NULL;
    n_x = n_y = n_z = NULL;
}

coVectField::coVectField(float *_x_in, float *_y_in, float *_z_in, float *_u_in, float *_v_in, float *_w_in, int _i_dim, int _j_dim, int _k_dim, float min_max_[6])
    : angle_((float)(9.5 * M_PI / 180.0))
    , cos_a_((float)cos(angle_))
    , sin_a_((float)sin(angle_))
    , cosenos_(0)
    , senos_(0)
    , arrow_factor_(0.20f)
    , i_dim(_i_dim)
    , j_dim(_j_dim)
    , k_dim(_k_dim)
    , x_in(_x_in)
    , y_in(_y_in)
    , z_in(_z_in)
    , u_in(_u_in)
    , v_in(_v_in)
    , w_in(_w_in)

{
    numc = i_dim * j_dim * k_dim;
    grdtype = UNI_GRD;
    for (int i = 0; i < 6; i++)
        min_max[i] = min_max_[i];
    s_out = NULL;
    num_scalar = 0;
    lines_out = NULL;
    u_scalar_data = NULL;
    s_in = NULL;
    n_x = n_y = n_z = NULL;
}

void coVectField::compute_vectorfields(float scale_, int length, int fasten_param, int num_sectors,
                                       float arrow_factor, float angle, const coObjInfo *objInfoLines, const coObjInfo *objInfoFloat, ValFlag flag)
{
    // dont change the original arrow settings
    float tmp[4];
    tmp[0] = arrow_factor_;
    tmp[1] = angle_;
    tmp[2] = cos_a_;
    tmp[3] = sin_a_;

    arrow_factor_ = arrow_factor;
    angle_ = (float)(angle * M_PI / 180.0);
    cos_a_ = cos(angle_);
    sin_a_ = sin(angle_);

    compute_vectorfields(scale_, length, fasten_param, num_sectors, objInfoLines, objInfoFloat, flag);

    arrow_factor_ = tmp[0];
    angle_ = tmp[1];
    cos_a_ = tmp[2];
    sin_a_ = tmp[3];
}
void coVectField::compute_vectorfields(float scale_, int length, int fasten_param, int num_sectors,
                                       const coObjInfo *objInfoLines, const coObjInfo *objInfoFloat, ValFlag flag)
{
    int i;

    scale = scale_;
    length_param = length;
    num_sectors_ = num_sectors;

#ifndef YAC
    if (objInfoLines && objInfoLines->getName())
#else
    if (objInfoLines)
#endif
    {
        if (num_sectors_)
        {
            cosenos_ = new float[num_sectors_];
            senos_ = new float[num_sectors_];
            int i;
            for (i = 0; i < num_sectors_; ++i)
            {
                cosenos_[i] = (float)cos((2.0 * M_PI * i) / num_sectors_);
                senos_[i] = (float)sin((2.0 * M_PI * i) / num_sectors_);
            }
        }
        else
        {
            cosenos_ = 0;
            senos_ = 0;
        }

        if (num_sectors_ >= 2)
            numc_line_2_ = 3 * (num_sectors_ / 2) + 2 * (num_sectors_ % 2);
        else if (num_sectors_ == 1)
            numc_line_2_ = 1;
        else
            numc_line_2_ = 0;

        if (num_sectors_ > 2)
        {
            numc_line_3_ = num_sectors_ - num_sectors_ % 2;
        }
        else
        {
            numc_line_3_ = 0;
        }

        lines_out = new coDoLines(*objInfoLines, numc * (2 + num_sectors_),
                                  numc * (2 + numc_line_2_ + numc_line_3_),
                                  numc);
        lines_out->getAddresses(&x_c, &y_c, &z_c, &v_l, &l_l);

        for (i = 0; i < numc; i++)
            l_l[i] = i * (2 + numc_line_2_ + numc_line_3_);

        if (!i_dim && !j_dim && !k_dim)
            create_lines();
        else if (grdtype == STR_GRD)
            create_strgrid_lines();
        else if (grdtype == RCT_GRD)
            create_rectgrid_lines();
        else if (grdtype == UNI_GRD)
            create_unigrid_lines(/*u_grid_in*/);

        if (n_x && n_y && n_z)
            project_lines(length_param == S_U);
        create_stings(); // sl: arrow points

        if (fasten_param == on_the_middle)
            vector_displacement();

        delete[] cosenos_;
        delete[] senos_;
    }
    else
    {
        lines_out = NULL;
    }

    num_scalar = numc * (2 + num_sectors_);
#ifndef YAC
    if (objInfoFloat && objInfoFloat->getName())
#else
    if (objInfoFloat)
#endif
    {
        u_scalar_data = new coDoFloat(*objInfoFloat, num_scalar);

        if (u_scalar_data->objectOk())
        {

            u_scalar_data->getAddress(&s_out);

            int const_delta = l_l[1] - l_l[0];
            for (i = 0; i < numc; i++)
                for (int j = l_l[i]; j < l_l[i] + const_delta; j++)
                    s_out[v_l[j]] = sqrt(u_in[v_l[l_l[i]] / 2] * u_in[v_l[l_l[i]] / 2] + v_in[v_l[l_l[i]] / 2] * v_in[v_l[l_l[i]] / 2] + w_in[v_l[l_l[i]] / 2] * w_in[v_l[l_l[i]] / 2]);
        }
    }
    else if (flag == PER_VERTEX)
    {
        s_out = new float[num_scalar];
        int const_delta = l_l[1] - l_l[0];
        for (i = 0; i < numc; i++)
            for (int j = l_l[i]; j < l_l[i] + const_delta; j++)
                s_out[v_l[j]] = sqrt(u_in[v_l[l_l[i]] / 2] * u_in[v_l[l_l[i]] / 2] + v_in[v_l[l_l[i]] / 2] * v_in[v_l[l_l[i]] / 2] + w_in[v_l[l_l[i]] / 2] * w_in[v_l[l_l[i]] / 2]);
    }
    else if (flag == PER_LINE)
    {
        s_out = new float[numc];
        for (i = 0; i < numc; i++)
            s_out[i] = sqrt(u_in[i] * u_in[i] + v_in[i] * v_in[i] + w_in[i] * w_in[i]);
    }
}

/*coVectField::coVectField(coDistributedObject *obj1, char *out_name1, coDistributedObject *obj2, char* out_name2, float Scale, int Length_param, int Fasten_param, int num_sectors_):
angle_(9.5*M_PI/180.0),cos_a_(cos(angle_)),sin_a_(sin(angle_)),
arrow_factor_(0.20),cosenos_(0),senos_(0),
scale(Scale), length_param(Length_param), fasten_param(Fasten_param)
{

DO_Structured_V3D_Data*	v_data_in=NULL;
coDoVec3*	uv_data_in=NULL;
coDoUniformGrid*		u_grid_in=NULL;
coDoRectilinearGrid*		r_grid_in=NULL;
coDoStructuredGrid*		s_grid_in=NULL;
coDoUnstructuredGrid*	        uns_grid_in=NULL;
coDoPolygons*		        poly_grid_in=NULL;
coDoPoints*			point_grid_in=NULL;
coDoTriangleStrips*		strips_grid_in=NULL;
coDoLines*			lines_grid_in=NULL;

lines_out = NULL;

int ivsize, jvsize, kvsize,vnumc=0;
int *dummy;

int i;

if (obj1->isType("STRGRD")) {
s_grid_in = (coDoStructuredGrid *)obj1;
s_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
s_grid_in->getAddresses(&x_in, &y_in, &z_in);
numc=i_dim*j_dim*k_dim;
}
else if (obj1->isType("UNSGRD")) {
uns_grid_in = (coDoUnstructuredGrid *)obj1;
uns_grid_in->getGridSize(&i_dim, &j_dim, &numc);
uns_grid_in->getAddresses(&dummy, &dummy, &x_in, &y_in, &z_in);
}
else if (obj1->isType("POLYGN")) {
poly_grid_in = (coDoPolygons *)obj1;
numc=poly_grid_in->getNumPoints();
poly_grid_in->getAddresses(&x_in, &y_in, &z_in, &dummy, &dummy);
}
else if (obj1->isType("POINTS")) {
point_grid_in = (coDoPoints *)obj1;
numc=point_grid_in->getNumPoints();
point_grid_in->getAddresses(&x_in, &y_in, &z_in);
}
else if (obj1->isType("TRIANG")) {
strips_grid_in = (coDoTriangleStrips *)obj1;
numc=strips_grid_in->getNumPoints();
strips_grid_in->getAddresses(&x_in, &y_in, &z_in, &dummy, &dummy);
}
else if (obj1->isType("LINES")) {
lines_grid_in = (coDoLines *)obj1;
numc=lines_grid_in->getNumPoints();
lines_grid_in->getAddresses(&x_in, &y_in, &z_in, &dummy, &dummy);
}
else if (obj1->isType("RCTGRD")) {
r_grid_in = (coDoRectilinearGrid *)obj1;
r_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
r_grid_in->getAddresses(&x_in, &y_in, &z_in);
numc=i_dim*j_dim*k_dim;
}
else if (obj1->isType("UNIGRD")) {
u_grid_in = (coDoUniformGrid *)obj1;
u_grid_in->getGridSize(&i_dim, &j_dim, &k_dim);
numc=i_dim*j_dim*k_dim;
}

if (obj2->isType("STRVDT")) {
v_data_in = (DO_Structured_V3D_Data *)obj2;
v_data_in->getGridSize(&ivsize, &jvsize, &kvsize);
vnumc=ivsize*jvsize*kvsize;
v_data_in->getAddresses(&u_in, &v_in, &w_in);
}
else if (obj2->isType("USTVDT")) {
uv_data_in = (coDoVec3 *)obj2;
vnumc = uv_data_in->getNumPoints();
uv_data_in->getAddresses(&u_in, &v_in, &w_in);
}

if(num_sectors_){
cosenos_ = new float[num_sectors_];
senos_ = new float[num_sectors_];
int i;
for(i=0;i<num_sectors_;++i){
cosenos_[i] = cos( (2.0 * M_PI * i) / num_sectors_ );
senos_[i] = sin( (2.0 * M_PI * i) / num_sectors_ );
}
}
else {
cosenos_ = 0;
senos_ = 0;
}

if(num_sectors_ >= 2)
numc_line_2_ = 3*(num_sectors_/2)+2*(num_sectors_%2) ;
else if(num_sectors_ == 1)
numc_line_2_ = 1;
else
numc_line_2_ = 0;

if(num_sectors_ > 2){
numc_line_3_ = num_sectors_ - num_sectors_%2;
}
else {
numc_line_3_ = 0;
}

lines_out = new coDoLines(out_name1,numc*(2+num_sectors_),
numc*(2+numc_line_2_+numc_line_3_),
numc);
if(!lines_out->objectOk()) {
delete [] cosenos_;
delete [] senos_;
}
lines_out->getAddresses(&x_c,&y_c,&z_c,&v_l,&l_l);

for(i=0;i<numc;i++)
l_l[i]=i*(2+numc_line_2_+numc_line_3_);

if (obj1->isType("STRGRD"))
create_strgrid_lines();
else if (obj1->isType("RCTGRD"))
create_rectgrid_lines();
else if (obj1->isType("UNIGRD"))
create_unigrid_lines(u_grid_in);
else
create_lines();

create_stings(); // sl: arrow points

if ( fasten_param == on_the_middle )
vector_displacement();

delete [] cosenos_;
delete [] senos_;
//p_outPort1->setCurrentObject(lines_out);

float *s_out;

u_scalar_data = new coDoFloat(out_name2, numc*(2+num_sectors_));

if (u_scalar_data->objectOk())
{

u_scalar_data->getAddress(&s_out);

int const_delta= l_l[1] - l_l[0];
for (i=0; i<lines_out->getNumLines(); i++)
for (int j=l_l[i]; j<l_l[i]+const_delta; j++){
s_out[v_l[j]] = sqrt(u_in[v_l[l_l[i]]/2]*u_in[v_l[l_l[i]]/2] + v_in[v_l[l_l[i]]/2]*v_in[v_l[l_l[i]]/2] + w_in[v_l[l_l[i]]/2]*w_in[v_l[l_l[i]]/2]);
}
}
}*/

void coVectField::create_strgrid_lines()
{
    int i, j, k, ijk;
    float len;

    for (i = 0; i < i_dim; i++)
        for (j = 0; j < j_dim; j++)
            for (k = 0; k < k_dim; k++)
            {
                ijk = i * j_dim * k_dim + j * k_dim + k;
                *(x_c + (ijk)*2) = *(x_in + ijk);
                *(y_c + (ijk)*2) = *(y_in + ijk);
                *(z_c + (ijk)*2) = *(z_in + ijk);

                switch (length_param)
                {
                case S_U:
                    len = sqrt(*(u_in + ijk) * *(u_in + ijk) + *(v_in + ijk) * *(v_in + ijk) + *(w_in + ijk) * *(w_in + ijk));
                    if (len > 0)
                    {
                        *(x_c + 1 + (ijk)*2) = *(x_in + ijk) + *(u_in + ijk) / len * scale;
                        *(y_c + 1 + (ijk)*2) = *(y_in + ijk) + *(v_in + ijk) / len * scale;
                        *(z_c + 1 + (ijk)*2) = *(z_in + ijk) + *(w_in + ijk) / len * scale;
                    }
                    else
                    {
                        *(x_c + 1 + (ijk)*2) = *(x_in + ijk);
                        *(y_c + 1 + (ijk)*2) = *(y_in + ijk);
                        *(z_c + 1 + (ijk)*2) = *(z_in + ijk);
                    }
                    break;
                case S_V:
                    *(x_c + 1 + (ijk)*2) = *(x_in + ijk) + *(u_in + ijk) * scale;
                    *(y_c + 1 + (ijk)*2) = *(y_in + ijk) + *(v_in + ijk) * scale;
                    *(z_c + 1 + (ijk)*2) = *(z_in + ijk) + *(w_in + ijk) * scale;
                    break;
                case S_DATA:
                    len = sqrt(*(u_in + ijk) * *(u_in + ijk) + *(v_in + ijk) * *(v_in + ijk) + *(w_in + ijk) * *(w_in + ijk));
                    if (len > 0)
                    {
                        *(x_c + 1 + (ijk)*2) = *(x_in + ijk) + *(u_in + ijk) / len * scale * (*(s_in + ijk));
                        *(y_c + 1 + (ijk)*2) = *(y_in + ijk) + *(v_in + ijk) / len * scale * (*(s_in + ijk));
                        *(z_c + 1 + (ijk)*2) = *(z_in + ijk) + *(w_in + ijk) / len * scale * (*(s_in + ijk));
                    }
                    else
                    {
                        *(x_c + 1 + (ijk)*2) = *(x_in + ijk);
                        *(y_c + 1 + (ijk)*2) = *(y_in + ijk);
                        *(z_c + 1 + (ijk)*2) = *(z_in + ijk);
                    }
                    break;
                }
            }
}

//======================================================================
// create the cutting planes
//======================================================================
void coVectField::create_rectgrid_lines()
{
    int i, j, k, ijk;
    float len;

    for (i = 0; i < i_dim; i++)
    {
        for (j = 0; j < j_dim; j++)
        {
            for (k = 0; k < k_dim; k++)
            {
                ijk = i * j_dim * k_dim + j * k_dim + k;
                *(x_c + (ijk)*2) = *(x_in + i);
                *(y_c + (ijk)*2) = *(y_in + j);
                *(z_c + (ijk)*2) = *(z_in + k);
                switch (length_param)
                {
                case S_U:
                    len = sqrt(*(u_in + ijk) * *(u_in + ijk) + *(v_in + ijk) * *(v_in + ijk) + *(w_in + ijk) * *(w_in + ijk));
                    if (len > 0)
                    {
                        *(x_c + 1 + (ijk)*2) = *(x_in + i) + *(u_in + ijk) / len * scale;
                        *(y_c + 1 + (ijk)*2) = *(y_in + j) + *(v_in + ijk) / len * scale;
                        *(z_c + 1 + (ijk)*2) = *(z_in + k) + *(w_in + ijk) / len * scale;
                    }
                    else
                    {
                        *(x_c + 1 + (ijk)*2) = *(x_in + ijk);
                        *(y_c + 1 + (ijk)*2) = *(y_in + ijk);
                        *(z_c + 1 + (ijk)*2) = *(z_in + ijk);
                    }
                    break;
                case S_V:
                    *(x_c + 1 + (ijk)*2) = *(x_in + i) + *(u_in + ijk) * scale;
                    *(y_c + 1 + (ijk)*2) = *(y_in + j) + *(v_in + ijk) * scale;
                    *(z_c + 1 + (ijk)*2) = *(z_in + k) + *(w_in + ijk) * scale;
                    break;
                case S_DATA:
                    len = sqrt(*(u_in + ijk) * *(u_in + ijk) + *(v_in + ijk) * *(v_in + ijk) + *(w_in + ijk) * *(w_in + ijk));
                    if (len > 0)
                    {
                        *(x_c + 1 + (ijk)*2) = *(x_in + i) + *(u_in + ijk) / len * scale * (*(s_in + ijk));
                        *(y_c + 1 + (ijk)*2) = *(y_in + j) + *(v_in + ijk) / len * scale * (*(s_in + ijk));
                        *(z_c + 1 + (ijk)*2) = *(z_in + k) + *(w_in + ijk) / len * scale * (*(s_in + ijk));
                    }
                    else
                    {
                        *(x_c + 1 + (ijk)*2) = *(x_in + ijk);
                        *(y_c + 1 + (ijk)*2) = *(y_in + ijk);
                        *(z_c + 1 + (ijk)*2) = *(z_in + ijk);
                    }
                    break;
                }
            }
        }
    }
}

//======================================================================
// create the cutting planes
//======================================================================
void coVectField::create_unigrid_lines()
{
    int i, j, k, ijk;
    float itmp, jtmp, ktmp, len;

    for (i = 0; i < i_dim; i++)
    {
        for (j = 0; j < j_dim; j++)
        {
            for (k = 0; k < k_dim; k++)
            {
                ijk = i * j_dim * k_dim + j * k_dim + k;
                if (i_dim > 1)
                    itmp = min_max[0] + ((float)i / (float)(i_dim - 1.0)) * (min_max[1] - min_max[0]);
                else
                    itmp = min_max[0];
                if (j_dim > 1)
                    jtmp = min_max[2] + ((float)j / (float)(j_dim - 1.0)) * (min_max[3] - min_max[2]);
                else
                    jtmp = min_max[2];
                if (j_dim > 1)
                    ktmp = min_max[4] + ((float)k / (float)(k_dim - 1.0)) * (min_max[5] - min_max[4]);
                else
                    ktmp = min_max[4];

                *(x_c + (ijk)*2) = itmp;
                *(y_c + (ijk)*2) = jtmp;
                *(z_c + (ijk)*2) = ktmp;

                switch (length_param)
                {
                case S_U:
                    len = sqrt(*(u_in + ijk) * *(u_in + ijk) + *(v_in + ijk) * *(v_in + ijk) + *(w_in + ijk) * *(w_in + ijk));
                    if (len > 0)
                    {
                        *(x_c + 1 + (ijk)*2) = itmp + *(u_in + ijk) / len * scale;
                        *(y_c + 1 + (ijk)*2) = jtmp + *(v_in + ijk) / len * scale;
                        *(z_c + 1 + (ijk)*2) = ktmp + *(w_in + ijk) / len * scale;
                    }
                    else
                    {
                        *(x_c + 1 + (ijk)*2) = itmp;
                        *(y_c + 1 + (ijk)*2) = jtmp;
                        *(z_c + 1 + (ijk)*2) = ktmp;
                    }
                    break;
                case S_V:
                    *(x_c + 1 + (ijk)*2) = itmp + *(u_in + ijk) * scale;
                    *(y_c + 1 + (ijk)*2) = jtmp + *(v_in + ijk) * scale;
                    *(z_c + 1 + (ijk)*2) = ktmp + *(w_in + ijk) * scale;
                    break;
                case S_DATA:
                    len = sqrt(*(u_in + ijk) * *(u_in + ijk) + *(v_in + ijk) * *(v_in + ijk) + *(w_in + ijk) * *(w_in + ijk));
                    if (len > 0)
                    {
                        *(x_c + 1 + (ijk)*2) = itmp + *(u_in + ijk) / len * scale * (*(s_in + ijk));
                        *(y_c + 1 + (ijk)*2) = jtmp + *(v_in + ijk) / len * scale * (*(s_in + ijk));
                        *(z_c + 1 + (ijk)*2) = ktmp + *(w_in + ijk) / len * scale * (*(s_in + ijk));
                    }
                    else
                    {
                        *(x_c + 1 + (ijk)*2) = itmp;
                        *(y_c + 1 + (ijk)*2) = jtmp;
                        *(z_c + 1 + (ijk)*2) = ktmp;
                    }
                    break;
                }
            }
        }
    }
}

//======================================================================
// create the cutting planes
//======================================================================
void coVectField::create_lines()
{
    int i;
    float len, *x_c_p, *y_c_p, *z_c_p, *x_in_p, *y_in_p, *z_in_p, *u_in_p, *v_in_p, *w_in_p, *s_in_p;

    x_c_p = x_c;
    y_c_p = y_c;
    z_c_p = z_c;
    x_in_p = x_in;
    y_in_p = y_in;
    z_in_p = z_in;
    u_in_p = u_in;
    v_in_p = v_in;
    w_in_p = w_in;
    s_in_p = s_in;
    for (i = 0; i < numc; i++)
    {
        *x_c_p++ = *x_in_p;
        *y_c_p++ = *y_in_p;
        *z_c_p++ = *z_in_p;

        switch (length_param)
        {
        case S_U:
            len = sqrt(*u_in_p * *u_in_p + *v_in_p * *v_in_p + *w_in_p * *w_in_p);
            if (len > 0)
            {
                *x_c_p++ = *x_in_p + *u_in_p / len * scale;
                *y_c_p++ = *y_in_p + *v_in_p / len * scale;
                *z_c_p++ = *z_in_p + *w_in_p / len * scale;
            }
            else
            {
                *x_c_p++ = *x_in_p;
                *y_c_p++ = *y_in_p;
                *z_c_p++ = *z_in_p;
            }
            break;
        case S_V:
            *x_c_p++ = *x_in_p + *u_in_p * scale;
            *y_c_p++ = *y_in_p + *v_in_p * scale;
            *z_c_p++ = *z_in_p + *w_in_p * scale;
            break;
        case S_DATA:
            len = sqrt(*(u_in + i) * *(u_in + i) + *(v_in + i) * *(v_in + i) + *(w_in + i) * *(w_in + i));
            if (len > 0)
            {
                *x_c_p++ = *x_in_p + *u_in_p / len * scale * (*s_in_p);
                *y_c_p++ = *y_in_p + *v_in_p / len * scale * (*s_in_p);
                *z_c_p++ = *z_in_p + *w_in_p / len * scale * (*s_in_p);
            }
            else
            {
                *x_c_p++ = *x_in_p;
                *y_c_p++ = *y_in_p;
                *z_c_p++ = *z_in_p;
            }
            break;
        }
        x_in_p++;
        y_in_p++;
        z_in_p++;
        u_in_p++;
        v_in_p++;
        w_in_p++;
        s_in_p++;
    }
}

void coVectField::project_lines(int keepLength)
{
    float n_len, old_len, new_len, s;
    float *x_start, *y_start, *z_start, *x_end, *y_end, *z_end,
        *x_normal, *y_normal, *z_normal;
    float dx, dy, dz;

    x_start = x_c;
    y_start = y_c;
    z_start = z_c;
    x_end = x_c + 1;
    y_end = y_c + 1;
    z_end = z_c + 1;
    x_normal = n_x;
    y_normal = n_y;
    z_normal = n_z;
    for (int i = 0; i < numc; i++)
    {
        n_len = sqrt(*x_normal * *x_normal + *y_normal * *y_normal + *z_normal * *z_normal);
        if (n_len > 0.0)
        {
            // normalize normal
            *x_normal /= n_len;
            *y_normal /= n_len;
            *z_normal /= n_len;
            // get difference (arrow-vector)
            dx = *x_end - *x_start;
            dy = *y_end - *y_start;
            dz = *z_end - *z_start;
            old_len = sqrt(dx * dx + dy * dy + dz * dz);
            // project arrow on normal: scalar product <arrow, normal>
            s = dx * *x_normal
                + dy * *y_normal
                + dz * *z_normal;
            // subtract normal component
            dx -= *x_normal * s;
            dy -= *y_normal * s;
            dz -= *z_normal * s;
            // reset length
            if (keepLength)
            {
                new_len = sqrt(dx * dx + dy * dy + dz * dz);
                if (new_len > 0)
                {
                    dx *= old_len / new_len;
                    dy *= old_len / new_len;
                    dz *= old_len / new_len;
                }
            }
            // calculate end point
            *x_end = *x_start + dx;
            *y_end = *y_start + dy;
            *z_end = *z_start + dz;
        }
        else
        {
            *x_end = *x_start;
            *y_end = *y_start;
            *z_end = *z_start;
        }
        x_start += 2;
        y_start += 2;
        z_start += 2;
        x_end += 2;
        y_end += 2;
        z_end += 2;
        x_normal++;
        y_normal++;
        z_normal++;
    }
}

// sl: the arrow point approximates a cone with a pyramid
void coVectField::create_stings()
{

    // Fill the points
    int i;
    for (i = 0; i < 2 * numc; i += 2)
    {
        fillTheStingPoints(i);
    }

    int begin_line;

    int numc_line = 2 + numc_line_2_ + numc_line_3_;
    for (i = 0; i < numc; ++i)
    {
        v_l[numc_line * i] = 2 * i;
        v_l[numc_line * i + 1] = 2 * i + 1;
    }
    begin_line = 2;
    int j;
    if (numc_line_2_)
    {
        for (i = 0; i < numc; ++i, begin_line += numc_line)
        {
            for (j = 0; j < num_sectors_ / 2; ++j)
            {
                v_l[begin_line + 3 * j + 0] = 2 * numc + num_sectors_ * i + 2 * j;
                v_l[begin_line + 3 * j + 1] = 2 * numc + num_sectors_ * i + 2 * j + 1;
                v_l[begin_line + 3 * j + 2] = 2 * i + 1;
            }
            if (num_sectors_ % 2)
            {
                if (num_sectors_ == 1)
                {
                    v_l[begin_line + 0] = 2 * numc + i;
                }
                else
                {
                    v_l[begin_line + 3 * j + 0] = 2 * numc + num_sectors_ * (i + 1) - 1;
                    v_l[begin_line + 3 * j + 1] = 2 * numc + num_sectors_ * i;
                }
            }
        }
    }

    begin_line = 2 + numc_line_2_;
    if (numc_line_3_)
        for (i = 0; i < numc; ++i, begin_line += numc_line)
        {
            for (j = 1; j < num_sectors_; ++j)
            {
                v_l[begin_line + j - 1] = 2 * numc + num_sectors_ * i + j;
            }
            if (num_sectors_ % 2 == 0)
            {
                v_l[begin_line + j - 1] = 2 * numc + num_sectors_ * i;
            }
        }
}

void coVectField::fillTheStingPoints(const int i)
{
    float v[3];
    float length;
    int j, k;

    int limit = 2 * numc + num_sectors_ * ((i / 2) + 1);

    // the vector
    v[0] = x_c[i + 1] - x_c[i];
    v[1] = y_c[i + 1] - y_c[i];
    v[2] = z_c[i + 1] - z_c[i];

    // load the points with the values of the arrow point...
    for (j = 2 * numc + num_sectors_ * (i / 2); j < limit; ++j)
    {
        x_c[j] = x_c[i + 1];
        y_c[j] = y_c[i + 1];
        z_c[j] = z_c[i + 1];
    }

    if (v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0)
        return;

    length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

    float v0[3], v1[3];

    // {v0, v1, v} are a direct base and v0, v1 are normalised
    orthoBase(v0, v1, v);

    v0[0] *= arrow_factor_ * length * sin_a_;
    v0[1] *= arrow_factor_ * length * sin_a_;
    v0[2] *= arrow_factor_ * length * sin_a_;
    v1[0] *= arrow_factor_ * length * sin_a_;
    v1[1] *= arrow_factor_ * length * sin_a_;
    v1[2] *= arrow_factor_ * length * sin_a_;

    // the points have been initially laden with the values of the arrow point...
    float x_shift0 = arrow_factor_ * cos_a_ * v[0];
    float y_shift0 = arrow_factor_ * cos_a_ * v[1];
    float z_shift0 = arrow_factor_ * cos_a_ * v[2];
    for (k = 0, j = 2 * numc + num_sectors_ * (i / 2); k < num_sectors_; ++j, ++k)
    {
        x_c[j] -= x_shift0;
        y_c[j] -= y_shift0;
        z_c[j] -= z_shift0;

        x_c[j] += cosenos_[k] * v0[0];
        y_c[j] += cosenos_[k] * v0[1];
        z_c[j] += cosenos_[k] * v0[2];

        x_c[j] += senos_[k] * v1[0];
        y_c[j] += senos_[k] * v1[1];
        z_c[j] += senos_[k] * v1[2];
    }
}

// assume v2 is not 0!!!!
void coVectField::orthoBase(float v0[3], float v1[3], const float v2[3])
{
    float length;

    if (fabs(v2[0]) <= fabs(v2[1]) && fabs(v2[0]) <= fabs(v2[2]))
    {
        // X is the smallest component
        v1[0] = 0.0;
        v1[1] = v2[2];
        v1[2] = -v2[1];
    }
    else if (fabs(v2[1]) <= fabs(v2[0]) && fabs(v2[1]) <= fabs(v2[2]))
    {
        // Y is the smallest component
        v1[0] = v2[2];
        v1[1] = 0.0;
        v1[2] = -v2[0];
    }
    else
    {
        // Z is the smallest component
        v1[0] = v2[1];
        v1[1] = -v2[0];
        v1[2] = 0.0;
    }

    length = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    v1[0] /= length;
    v1[1] /= length;
    v1[2] /= length;

    v0[0] = v1[1] * v2[2] - v1[2] * v2[1];
    v0[1] = v1[2] * v2[0] - v1[0] * v2[2];
    v0[2] = v1[0] * v2[1] - v1[1] * v2[0];

    length = sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
    v0[0] /= length;
    v0[1] /= length;
    v0[2] /= length;
}

//===================================================================
// vector_displacement ,the vectors will be fastened on the middle,
// 19.08.98
//===================================================================
void coVectField::vector_displacement()
{
    int ijk;
    int j, k;
    float x_disp, y_disp, z_disp;
    for (ijk = 0; ijk < numc; ijk++)
    {
        x_disp = (*(x_c + 1 + (ijk)*2) - *(x_c + (ijk)*2)) / 2;
        y_disp = (*(y_c + 1 + (ijk)*2) - *(y_c + (ijk)*2)) / 2;
        z_disp = (*(z_c + 1 + (ijk)*2) - *(z_c + (ijk)*2)) / 2;

        *(x_c + (ijk)*2) = *(x_c + (ijk)*2) - x_disp;
        *(y_c + (ijk)*2) = *(y_c + (ijk)*2) - y_disp;
        *(z_c + (ijk)*2) = *(z_c + (ijk)*2) - z_disp;

        *(x_c + 1 + (ijk)*2) = *(x_c + 1 + (ijk)*2) - x_disp;
        *(y_c + 1 + (ijk)*2) = *(y_c + 1 + (ijk)*2) - y_disp;
        *(z_c + 1 + (ijk)*2) = *(z_c + 1 + (ijk)*2) - z_disp;

        for (k = 0, j = 2 * numc + num_sectors_ * ijk; k < num_sectors_; ++j, ++k)
        {
            x_c[j] -= x_disp;
            y_c[j] -= y_disp;
            z_c[j] -= z_disp;
        }
    }
}

#ifndef YAC
coDistrVectField::coDistrVectField(const coDistributedObject *geo,
                                   const coDistributedObject *vect,
                                   const coDoColormap *colorMap,
                                   float scale, int lineChoice, int numsectors, int projectlines)
    : _geo(geo)
    , _vect(vect)
    , _colorMap(colorMap)
    , _scale(scale)
    , _length_param(lineChoice)
    , _fasten_param(0)
    , _num_sectors(numsectors)
    , _project_lines(projectlines)
{
    _arrow_factor = 0.20f;
    _angle = 9.5f;
}

coDistrVectField::coDistrVectField(const coDistributedObject *geo,
                                   const coDistributedObject *vect,
                                   const coDoColormap *colorMap,
                                   float scale, int lineChoice, int numsectors, float arrow_factor,
                                   float angle, int projectlines)
    : _geo(geo)
    , _vect(vect)
    , _colorMap(colorMap)
    , _scale(scale)
    , _length_param(lineChoice)
    , _fasten_param(0)
    , _num_sectors(numsectors)
    , _arrow_factor(arrow_factor)
    , _angle(angle)
    , _project_lines(projectlines)
{
}

namespace covise
{

static void
RedressVertexOrder(const coDistributedObject *lines, const coDistributedObject *colorLines)
{
    if (lines == NULL || colorLines == NULL)
    {
        return; // nothing to redress
    }
    if (dynamic_cast<const coDoSet *>(lines) && dynamic_cast<const coDoSet *>(colorLines))
    {
        const coDoSet *Lines = dynamic_cast<const coDoSet *>(lines);
        const coDoSet *ColorLines = dynamic_cast<const coDoSet *>(colorLines);
        int no_lines;
        const coDistributedObject *const *lineList = Lines->getAllElements(&no_lines);
        int no_colors;
        const coDistributedObject *const *colorList = ColorLines->getAllElements(&no_colors);
        if (no_colors == no_lines)
        {
            int obj;
            for (obj = 0; obj < no_lines; ++obj)
            {
                RedressVertexOrder(lineList[obj], colorList[obj]);
            }
        }
        int obj;
        for (obj = 0; obj < no_lines; ++obj)
        {
            delete lineList[obj];
        }
        for (obj = 0; obj < no_colors; ++obj)
        {
            delete colorList[obj];
        }
        delete[] lineList;
        delete[] colorList;
    }
    else if (dynamic_cast<const coDoLines *>(lines) && dynamic_cast<const coDoRGBA *>(colorLines))
    {
        const coDoLines *Lines = dynamic_cast<const coDoLines *>(lines);
        const coDoRGBA *ColorLines = dynamic_cast<const coDoRGBA *>(colorLines);
        int no_l = Lines->getNumLines();
        int no_v = Lines->getNumVertices();
        int no_p = Lines->getNumPoints();
        int *l_l, *v_l;
        float *x, *y, *z;
        Lines->getAddresses(&x, &y, &z, &v_l, &l_l);
        int no_col = ColorLines->getNumPoints();
        if (no_l == 0)
        {
            return;
        }
        else if (no_p % no_l != 0 || no_v % no_l != 0 || no_p < no_l || no_col != no_p)
        {
            cerr << "RedressVertexOrder in coVectField: wrong data for these lines" << endl;
            return;
        }
        int *tmp = new int[no_col];
        int *colors;
        ColorLines->getAddress(&colors);
        memcpy(tmp, colors, no_col * sizeof(int));
        int vertex;
        int linea;
        int repeat_p = no_p / no_l;
        int repeat_v = no_v / no_l;
        for (linea = 0; linea < no_l; ++linea)
        {
            int color = tmp[repeat_p * linea];
            //int num_vertices;
            for (vertex = 0; vertex < repeat_v; ++vertex)
            {
                colors[v_l[repeat_v * linea + vertex]] = color;
            }
        }
        delete[] tmp;
    }
}
}

bool
coDistrVectField::Execute(coDistributedObject **lines,
                          coDistributedObject **colorSurf,
                          coDistributedObject **colorLines,
                          string linesName,
                          string colorName,
                          bool ColorMapAttrib,
                          const ScalarContainer *SCont,
                          int vectOpt)
{
    *lines = NULL;
    *colorSurf = NULL;
    *colorLines = NULL;

    if (_geo == NULL || _vect == NULL)
    {
        return false;
    }

    ScalarContainer scalar;
    bool ok = CreateLinesAndScalar(lines, linesName,
                                   _geo, _vect, scalar);
    if (!ok)
    {
        *lines = NULL;
        return false;
    }
    if (vectOpt < 2) // surface colors
    {
        ok = CreateColors(colorSurf, string(colorName + "_Surf"),
                          scalar, ColorMapAttrib, SCont, 1);
        if (!ok)
        {
            return false;
        }
    }
    if (vectOpt != 1) //line colors
    {
        ok = CreateColors(colorLines, string(colorName + "_Lines"),
                          scalar, ColorMapAttrib, SCont, _num_sectors + 2);
        // unluckily the way vertex are organised in *lines is
        // not the one assumed in CreateColors...
        RedressVertexOrder(*lines, *colorLines); // quick and dirty solution
        if (!ok)
        {
            return false;
        }
    }
    return true;
}

bool
coDistrVectField::CreateColors(coDistributedObject **color,
                               string colorName,
                               const ScalarContainer &scalar,
                               bool createCMAP, const ScalarContainer *SCont,
                               int repeat)
{
    coColors colors(scalar, _colorMap, false, SCont);
    *color = colors.getColors(colorName.c_str(), false, createCMAP, repeat);
    return true;
}

bool
coDistrVectField::CreateLinesAndScalar(coDistributedObject **lines,
                                       string linesName,
                                       const coDistributedObject *geo,
                                       const coDistributedObject *vect,
                                       ScalarContainer &scalar)
{
    float *xin = NULL;
    float *yin = NULL;
    float *zin = NULL;
    float *uin = NULL;
    float *vin = NULL;
    float *win = NULL;
    int *vl = NULL, *ll = NULL;
    int no_elem, no_vert, no_corn, vect_no_of_points = 0;

    *lines = NULL;

    if (dynamic_cast<const coDoSet *>(geo) && dynamic_cast<const coDoSet *>(vect))
    {
        int no_e;
        const coDistributedObject *const *geoms = ((coDoSet *)(geo))->getAllElements(&no_e);
        int no_d;
        const coDistributedObject *const *vects = ((coDoSet *)(vect))->getAllElements(&no_d);
        if (no_e != no_d)
        {
            return false;
        }

        int i;
        coDistributedObject **setList = new coDistributedObject *[no_e + 1];
        scalar.OpenList(no_e);
        scalar.CopyAllAttributes(vect);
        setList[no_e] = NULL;
        for (i = 0; i < no_e; ++i)
        {
            setList[i] = NULL;
            string name_i = linesName;
            char buf[16];
            sprintf(buf, "_%d", i);
            name_i += buf;
            if (linesName == "")
            {
                name_i = "";
            }
            bool ok = CreateLinesAndScalar(&setList[i], name_i, geoms[i], vects[i], scalar[i]);
            if (!ok)
            {
                return false;
            }
        }
        if (linesName == "")
        {
            *lines = NULL;
        }
        else
        {
            // FIXME
            *lines = new coDoSet(linesName.c_str(), setList);
        }
        for (i = 0; i < no_e; ++i)
        {
            delete geoms[i];
            delete vects[i];
            delete setList[i];
        }
        delete[] setList;
        delete[] geoms;
        delete[] vects;
        return true;
    }
    else if (dynamic_cast<const coDoPolygons *>(geo) && dynamic_cast<const coDoVec3 *>(vect))
    {
        const coDoPolygons *Geo = (coDoPolygons *)geo;
        no_elem = Geo->getNumPolygons();
        no_vert = Geo->getNumVertices();
        no_corn = Geo->getNumPoints();
        Geo->getAddresses(&xin, &yin, &zin, &vl, &ll);

        const coDoVec3 *Vect = (coDoVec3 *)vect;
        Vect->getAddresses(&uin, &vin, &win);
        vect_no_of_points = Vect->getNumPoints();
    }
    else if (dynamic_cast<const coDoTriangleStrips *>(geo) && dynamic_cast<const coDoVec3 *>(vect))
    {
        const coDoTriangleStrips *Geo = (coDoTriangleStrips *)geo;
        no_elem = Geo->getNumStrips();
        no_vert = Geo->getNumVertices();
        no_corn = Geo->getNumPoints();
        Geo->getAddresses(&xin, &yin, &zin, &vl, &ll);

        const coDoVec3 *Vect = (coDoVec3 *)vect;
        Vect->getAddresses(&uin, &vin, &win);
        vect_no_of_points = Vect->getNumPoints();
    }
    else
    {
        return false;
    }

    if (no_corn == vect_no_of_points)
    {
        coObjInfo linesInfo(linesName);
        coVectField auxVectField(no_corn, xin, yin, zin, uin, vin, win);

        // calculate normal list (can handle "TRIANG" and "POLYGN")
        float *normal_x = NULL, *normal_y = NULL, *normal_z = NULL;
        if (_project_lines)
        {
            normal_x = new float[no_corn];
            normal_y = new float[no_corn];
            normal_z = new float[no_corn];
            float n_x, n_y, n_z;
            int point0, point1, point2, v_start, v_end;
            for (int elem = 0; elem < no_elem; elem++)
            {
                v_start = ll[elem];
                if (elem < no_corn - 1)
                    v_end = ll[elem + 1];
                else
                    v_end = no_vert;
                for (int i = v_start; i < v_end; i++)
                {
                    // calculate normal per point (nescessary for triangle strips)
                    point1 = vl[i];
                    if (i == v_start)
                    {
                        point0 = vl[i + 1];
                        point2 = vl[i + 2];
                    }
                    else if (i == v_end - 1)
                    {
                        point0 = vl[i - 1];
                        point2 = vl[i - 2];
                    }
                    else
                    {
                        point0 = vl[i - 1];
                        point2 = vl[i + 1];
                    }
                    n_x = (yin[point1] - yin[point0]) * (zin[point2] - zin[point0]) - (zin[point1] - zin[point0]) * (yin[point2] - yin[point0]);
                    n_y = (zin[point1] - zin[point0]) * (xin[point2] - xin[point0]) - (xin[point1] - xin[point0]) * (zin[point2] - zin[point0]);
                    n_z = (xin[point1] - xin[point0]) * (yin[point2] - yin[point0]) - (yin[point1] - yin[point0]) * (xin[point2] - xin[point0]);
                    // set normal
                    normal_x[point1] = n_x;
                    normal_y[point1] = n_y;
                    normal_z[point1] = n_z;
                }
            }
            auxVectField.setProjectionNormals(normal_x, normal_y, normal_z);
        }

        auxVectField.compute_vectorfields(_scale, _length_param, _fasten_param,
                                          _num_sectors, _arrow_factor, _angle, &linesInfo, NULL,
                                          coVectField::PER_LINE);

        delete[] normal_x;
        delete[] normal_y;
        delete[] normal_z;

        *lines = auxVectField.get_obj_lines();
        scalar.AddArray(no_corn, auxVectField.get_scalar_data());
        scalar.CopyAllAttributes(vect);
        return true;
    }
    *lines = new coDoLines(linesName.c_str(), 0, 0, 0);
    scalar.CopyAllAttributes(vect);
    return true;
}
#endif // YAC
