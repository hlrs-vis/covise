#ifndef __SHAREDTYPES__
#define __SHAREDTYPES__

#include <vector>
#include <map>
#include <string>

#include <Dense>
#include <StdVector>
#include "sigslot.h"

using namespace Eigen;

#define DHINDEX(x) x-1
#define IN
#define OUT
#define FMACRO(axis_one , axis_two) (axis_one + nu*axis_two)/(1 + nu)

enum JointT {REVOLUTE , PRISMATIC , CONSTANTJOINT , NOTSET};
enum AxisT  {AxisX , AxisY , AxisZ};

struct dh_parametrs
{
    float a;            //Length of common normal
    float alpha;        //Angle between zi and zi-1 along xi
    float d;            //distance between xi and xi-1 along zi (variable for prismatic)
    float theta;        //Angle between xi and xi-1 along zi    (variavle for revolute)
    JointT z_joint_type;//Joint type at z-1 axis
    std::string joint_name;
};

typedef std::vector<dh_parametrs> dh_table;
typedef std::vector<Vector3f , aligned_allocator<Vector3f> > vector_storage;
typedef std::vector<Matrix4f , aligned_allocator<Matrix4f> > HomMatrixHolder;

#endif