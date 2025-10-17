#version 420 core
INCLUDE vis_surfel_util.glsl

layout(location = 0)  in vec3  in_position;
layout(location = 1)  in float in_r;
layout(location = 2)  in float in_g;
layout(location = 3)  in float in_b;
layout(location = 4)  in float empty;
layout(location = 5)  in float in_radius;   // Roh-RADIUS (Attribut aus Vorverarbeitung)
layout(location = 6)  in vec3  in_normal;

uniform mat4  mvp_matrix;

uniform float min_radius;          // World-CLAMP (Radius)
uniform float max_radius;          // World-CLAMP (Radius)
uniform float scale_radius;        // Roh-Radius -> Welt-Radius
uniform float scale_radius_gamma;  // Gamma auf Roh-Durchmesser
uniform float max_radius_cut;      // CUT im Roh-RADIUS (vor Scaling!)

out VertexData {
    vec3  pass_ms_u;         // Halbachse U (WS, Radius)
    vec3  pass_ms_v;         // Halbachse V (WS, Radius)
    vec3  pass_point_color;  // RGB
    vec3  pass_world_pos;    // Mittelpunkt (WS)
    float pass_radius_ws;    // Durchmesser (WS) nach World-CLAMP (Pixel-CLAMP folgt im GS)
} VertexOut;

void main() {
    const float EPS = 1e-6;

    // Map raw radius to world radius
    float r_ws = map_raw_to_world_radius(
        in_radius,
        max_radius_cut,
        scale_radius_gamma,
        scale_radius,
        min_radius,
        max_radius);

    // Orthonormal basis from normal
    vec3 u, v;
    onb_from_normal(in_normal, u, v);

    VertexOut.pass_ms_u        = u * r_ws;
    VertexOut.pass_ms_v        = v * r_ws;
    VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
    VertexOut.pass_world_pos   = in_position;
    VertexOut.pass_radius_ws   = r_ws;

    gl_Position = mvp_matrix * vec4(in_position, 1.0);
}
