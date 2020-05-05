#include "CLineFactory.h"
#include <math.h>

void CLine3D::FillTraectory( IN vector_storage & in_vec )
{
    traectory = in_vec;
}


//Ax + By + Cz + D = 0 - equation for line in 3d

Vector3f CLine3D::GetNextPoint( IN const Vector3f & point_end , IN const Vector3f & point_start, IN const float & displacement )
{
    Vector3f next_point;
    float nu = 0;
    
    float r1 = point_end[0] - point_start[0];
    float r2 = point_end[1] - point_start[1];
    float r3 = point_end[2] - point_start[2];
    
    //calculate length of vector
    float vec_length = sqrt(r1*r1 + r2*r2 + r3*r3);
    //get number of points with such displacement
    unsigned int number_of_points = (unsigned int)(vec_length/displacement);

    if(number_of_points == 1)
        //TODO check it. maybe i am wrong
        //This is point i-1, where i - last point. So next point is point end.
        return point_end;
    else
        nu = static_cast<float>(1.0f/(number_of_points-1));
    //Calculate next point
    float x_new = FMACRO(point_start[0] , point_end[0]);
    float y_new = FMACRO(point_start[1] , point_end[1]);
    float z_new = FMACRO(point_start[2] , point_end[2]);
    //Fill vector
    next_point <<   x_new , //x
                    y_new , //y
                    z_new ; //z

    return next_point;
}