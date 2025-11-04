#version 420 core

INCLUDE vis_point_util.glsl

out VertexData {
    vec3 color;
} VertexOut;

layout(location = 0) in vec3  in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 5) in float in_radius;

uniform mat4  mvp_matrix;
uniform float min_screen_size;
uniform float max_screen_size;
uniform float min_radius;
uniform float max_radius;
uniform float scale_radius; 
uniform float scale_projection;
uniform float scale_radius_gamma;
uniform float max_radius_cut;
uniform bool  use_aniso;
uniform vec4  Pcol0;
uniform vec4  Pcol1;
uniform float viewport_half_y;
uniform float aniso_normalize;

void main() {
    const float EPS = 1e-6;

    float r_ws = calc_world_radius(
        in_radius, max_radius_cut, scale_radius_gamma, scale_radius, min_radius, max_radius);

    vec4 clip = mvp_matrix * vec4(in_position, 1.0);
    VertexOut.color = vec3(in_r, in_g, in_b);

    if (abs(clip.w) <= EPS || r_ws <= UTIL_EPS) {
        gl_Position = clip;
        gl_PointSize = 0.0;
        return;
    }

    float pointSize = use_aniso
        ? calc_diameter_px_aniso(clip, r_ws, Pcol0, Pcol1, viewport_half_y, aniso_normalize,
                                 min_screen_size, max_screen_size)
        : calc_diameter_px_iso(clip, r_ws, scale_projection, min_screen_size, max_screen_size);

    gl_Position = clip;
    gl_PointSize = (pointSize > UTIL_EPS) ? pointSize : 0.0;
}
