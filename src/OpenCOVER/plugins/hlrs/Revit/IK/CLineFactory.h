#ifndef __CLINEFACTORY__
#define __CLINEFACTORY__

#include <Dense>
#include <vector>
#include "SharedTypes.h"

using namespace Eigen;

class CLine3D
{
    vector_storage traectory;
public:
    CLine3D(){}

    Vector3f GetNextPoint(
                            IN const Vector3f & point_end ,
                            IN const Vector3f & point_start,
                            IN const float    & displacement
                         );

    void     FillTraectory(
                            IN vector_storage & in_vec
                          );
};

#endif