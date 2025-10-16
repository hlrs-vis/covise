// ===================== Vertex Shader =====================
#version 420 core

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

uniform float min_radius;         // Welt-CLAMP (Durchmesser)
uniform float max_radius;         // Welt-CLAMP (Durchmesser)
uniform float scale_radius;       // Roh -> Welt (Durchmesser)
uniform float scale_radius_gamma; // Gamma-Shaping (auf Roh-Durchmesser)
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

    // --- RAW-Radius & Cut im RAW-Domain ---
    float r_raw = max(0.0, in_radius);
    bool cut = (max_radius_cut > 0.0) && (r_raw > max_radius_cut);

    // --- WS-Radius mit Gamma/Scale, danach CLAMP im WS ---
    float r_ws = 0.0;
    if (!cut) {
        float gamma = (scale_radius_gamma > 0.0) ? scale_radius_gamma : 1.0;
        float r_ws_unclamped = scale_radius * pow(r_raw, gamma); // WS-Radius
        r_ws = clamp(r_ws_unclamped, min_radius, max_radius);    // << CLAMP NACH Skalierung (WS)
    }

    // Normale (WS/VS)
    vec3 n_ws = normalize(in_normal);
    vec3 n_vs = normalize(normal_matrix * in_normal);
    if (length(n_vs) < EPS) n_vs = vec3(0,0,1);

    // Orthonormale Tangentenbasis (WS)
    vec3 ref = (abs(n_ws.x) > abs(n_ws.y) && abs(n_ws.x) > abs(n_ws.z))
             ? vec3(0.0, 1.0, 0.0)
             : (abs(n_ws.y) > abs(n_ws.z) ? vec3(0.0, 0.0, 1.0)
                                          : vec3(1.0, 0.0, 0.0));
    vec3 u = normalize(cross(ref, n_ws));
    vec3 v = normalize(cross(n_ws, u));


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
