#ifndef RPLYCALLBACKS
#define RPLYCALLBACKS

#include "ColorSphere.h"

int readVertexCallback(p_ply_argument argument)
{
    long axis;
    void* pointer;
    ply_get_argument_user_data(argument, &pointer, &axis);
    point_vector* points = static_cast<point_vector*>(pointer);
    float val = ply_get_argument_value(argument);
    //std::cout << "read vertex  axis: " << axis << "  value: " << val << std::endl;

    static unsigned long count = 0;
    switch (axis)
    {
        case 0:
            points->at(count).center().x = val;
            break;
        case 1:
            points->at(count).center().y = val;
            break;
        case 2:
            points->at(count).center().z = val;
            count++;
            break;
    }

    return 1;
}

int readColorCallback(p_ply_argument argument)
{
    long axis;
    void* pointer;
    ply_get_argument_user_data(argument, &pointer, &axis);
    point_vector* points = static_cast<point_vector*>(pointer);

    int val = ply_get_argument_value(argument);
    //std::cout << "read color  axis: " << axis << "  value: " << val << std::endl;

    static unsigned long count = 0;
    switch (axis)
    {
        case 0:
            points->at(count).color().x = val / 255.0f;
            break;
        case 1:
            points->at(count).color().y = val / 255.0f;
            break;
        case 2:
            points->at(count).color().z = val / 255.0f;
            count++;
            break;
    }

    return 1;
}
#endif // RPLYCALLBACKS

