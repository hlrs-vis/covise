#version 420 core
INCLUDE vis_surfel_util.glsl

layout(location = 0)  in vec3  in_position;
layout(location = 1)  in float in_r;
layout(location = 2)  in float in_g;
layout(location = 3)  in float in_b;
layout(location = 4)  in float empty;
layout(location = 5)  in float in_radius;
layout(location = 6)  in vec3  in_normal;

layout(location = 7)  in float in_prov1;
layout(location = 8)  in float in_prov2;
layout(location = 9)  in float in_prov3;
layout(location = 10) in float in_prov4;
layout(location = 11) in float in_prov5;
layout(location = 12) in float in_prov6;

uniform mat4  mvp_matrix;
uniform float max_radius;
uniform float min_radius;
uniform float scale_radius;
uniform float max_radius_cut;
uniform float scale_radius_gamma;

out VertexData {
    vec3  pass_ms_u;        // Tangenten-Halbachse U (WS, = Radius)
    vec3  pass_ms_v;        // Tangenten-Halbachse V (WS, = Radius)
    vec3  pass_point_color;
    vec3  pass_world_pos;
    vec3  pass_normal_ws;
    float pass_radius_ws;   // Radius (WS, geclamped in WS-Grenzen)
    float pass_prov1;
    float pass_prov2;
    float pass_prov3;
    float pass_prov4;
    float pass_prov5;
    float pass_prov6;
} VertexOut;

void main() {
    vec3 n = normalize(in_normal);
    vec3 ref = (abs(n.x) > abs(n.y) && abs(n.x) > abs(n.z)) ? vec3(0.0,1.0,0.0)
              : (abs(n.y) > abs(n.z))                        ? vec3(0.0,0.0,1.0)
                                                             : vec3(1.0,0.0,0.0);
    vec3 u = normalize(cross(ref, n));
    vec3 v = normalize(cross(n, u));

    float r_world = map_raw_to_world_radius(
        in_radius,
        max_radius_cut,
        scale_radius_gamma,
        scale_radius,
        min_radius,
        max_radius
    ); // Radius (WS)

    VertexOut.pass_ms_u        = u * r_world;
    VertexOut.pass_ms_v        = v * r_world;
    VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
    VertexOut.pass_world_pos   = in_position;
    VertexOut.pass_normal_ws   = n;
    VertexOut.pass_radius_ws   = r_world;

    VertexOut.pass_prov1 = in_prov1;
    VertexOut.pass_prov2 = in_prov2;
    VertexOut.pass_prov3 = in_prov3;
    VertexOut.pass_prov4 = in_prov4;
    VertexOut.pass_prov5 = in_prov5;
    VertexOut.pass_prov6 = in_prov6;

    gl_Position = mvp_matrix * vec4(in_position, 1.0);
}
