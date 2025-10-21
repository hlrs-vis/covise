// ===================== Vertex Shader =====================
#version 420 core
INCLUDE vis_surfel_util.glsl

layout(location = 0)  in vec3  in_position;
layout(location = 1)  in float in_r;
layout(location = 2)  in float in_g;
layout(location = 3)  in float in_b;
layout(location = 4)  in float empty;
layout(location = 5)  in float in_radius;
layout(location = 6)  in vec3  in_normal;

uniform mat4  mvp_matrix;
uniform mat4  view_matrix;
uniform mat3  normal_matrix;

uniform float min_radius;         // Welt-CLAMP (Radius)
uniform float max_radius;         // Welt-CLAMP (Radius)
uniform float scale_radius;       // Roh -> Welt (Radius)
uniform float scale_radius_gamma; // Gamma-Shaping (auf Roh-Radius)
uniform float max_radius_cut;     // CUT-Schwelle im Roh-RADIUS (vor Scaling!)

out VertexData {
    vec3  pass_ms_u;         // Halbachse U (WS, Radius)
    vec3  pass_ms_v;         // Halbachse V (WS, Radius)
    vec3  pass_point_color;  // RGB
    vec3  pass_world_pos;    // Weltposition
    vec3  pass_normal_ws;    // Welt-Normale
    vec3  pass_vs_pos;       // View-Space Position
    vec3  pass_vs_normal;    // View-Space Normale
    float pass_radius_ws;    // Durchmesser (WS) nach Welt-CLAMP (0 bei Cut)
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

    // Normale (WS/VS)
    vec3 n_ws = normalize(in_normal);
    vec3 n_vs = normalize(normal_matrix * in_normal);
    if (length(n_vs) < EPS) n_vs = vec3(0,0,1);

    // Orthonormalbasis (Frisvad)
    vec3 u, v;
    onb_frisvad(n_ws, u, v);


    // Outputs
    VertexOut.pass_ms_u        = u * r_ws;
    VertexOut.pass_ms_v        = v * r_ws;
    VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
    VertexOut.pass_world_pos   = in_position;
    VertexOut.pass_normal_ws   = n_ws;
    VertexOut.pass_vs_pos      = (view_matrix * vec4(in_position, 1.0)).xyz;
    VertexOut.pass_vs_normal   = n_vs;
    VertexOut.pass_radius_ws   = r_ws; // Durchmesser (0 bei Cut)

    gl_Position = mvp_matrix * vec4(in_position, 1.0);
}
