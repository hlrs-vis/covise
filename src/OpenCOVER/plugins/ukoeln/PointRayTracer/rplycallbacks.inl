#ifndef RPLYCALLBACKS
#define RPLYCALLBACKS

#include "ColorSphere.h"

struct PLYCallbackData {
    point_vector* points;
    unsigned long vertex_count;
    unsigned long color_count;
};

int readVertexCallback(p_ply_argument argument)
{
    long axis;
    void* pointer;
    ply_get_argument_user_data(argument, &pointer, &axis);
    PLYCallbackData* data = static_cast<PLYCallbackData*>(pointer);
    float val = ply_get_argument_value(argument);

    switch (axis)
    {
        case 0:
            data->points->at(data->vertex_count).center().x = val;
            break;
        case 1:
            data->points->at(data->vertex_count).center().y = val;
            break;
        case 2:
            data->points->at(data->vertex_count).center().z = val;
            data->vertex_count++;
            break;
    }

    return 1;
}

int readColorCallback(p_ply_argument argument)
{
    long axis;
    void* pointer;
    ply_get_argument_user_data(argument, &pointer, &axis);
    PLYCallbackData* data = static_cast<PLYCallbackData*>(pointer);

    int val = ply_get_argument_value(argument);

    switch (axis)
    {
        case 0:
            data->points->at(data->color_count).color().x = val / 255.0f;
            break;
        case 1:
            data->points->at(data->color_count).color().y = val / 255.0f;
            break;
        case 2:
            data->points->at(data->color_count).color().z = val / 255.0f;
            data->color_count++;
            break;
    }

    return 1;
}
#endif // RPLYCALLBACKS

