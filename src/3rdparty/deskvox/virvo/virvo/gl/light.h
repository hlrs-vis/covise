
#ifndef VV_LIGHT_H
#define VV_LIGHT_H

#include "math/math.h"


namespace virvo
{
namespace gl
{

// https://www.opengl.org/sdk/docs/man2/xhtml/glGetLight.xml
struct light
{

    light()
        : ambient(vec4(0.0f, 0.0f, 0.0f, 1.0f))
        , diffuse(vec4(0.0f, 0.0f, 0.0f, 0.0f))
        , specular(vec4(0.0f, 0.0f, 0.0f, 0.0f))
        , position(vec4(0.0f, 0.0f, 1.0f, 0.0f))
        , spot_direction(vec3(0.0f, 0.0f, -1.0f))
        , spot_exponent(0.0f)
        , spot_cutoff(180.0f)
        , constant_attenuation(1.0f)
        , linear_attenuation(0.0f)
        , quadratic_attenuation(0.0f)
    {
    }

    vec4 ambient;
    vec4 diffuse;
    vec4 specular;

    vec4 position;

    vec3 spot_direction;
    float spot_exponent;
    float spot_cutoff;

    float constant_attenuation;
    float linear_attenuation;
    float quadratic_attenuation;

};

} // gl
} // virvo

#endif // VV_LIGHT_H


