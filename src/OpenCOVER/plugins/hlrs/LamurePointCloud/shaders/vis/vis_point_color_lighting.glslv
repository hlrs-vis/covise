#version 420 core

layout(location = 0) in vec3  in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 4) in float empty;
layout(location = 5) in float in_radius;
layout(location = 6) in vec3  in_normal;

uniform mat4 mvp_matrix;
uniform mat4 view_matrix;
uniform mat3 normal_matrix;

uniform float min_radius;        // world-space clamp (Radius)
uniform float max_radius;        // world-space clamp (Radius)
uniform float min_screen_size;     // screenspace clamp (Pixel)
uniform float max_screen_size;     // screenspace clamp (Pixel)
uniform float scale_radius;      // Roh -> Weltmaß
uniform float scale_projection;  // Weltmaß -> Pixel (unter Nutzung von |w|)
uniform float scale_radius_gamma; // Shaping-Gamma (z.B. 0.5)
uniform float max_radius_cut;    // Weltmaß-Cut (wirkt VOR Skalierung)

out VertexData {
    vec3  pass_point_color;
    vec3  pass_world_pos;
    vec3  pass_vs_pos;
    vec3  pass_vs_normal;
    float pass_radius_ws;    // nach Welt-Clamp
    float pass_screen_size;  // nach Pixel-Clamp
} VertexOut;

void main() {
    const float EPS = 1e-6;

    // Positionen & Normalen
    vec4 worldPos4 = vec4(in_position, 1.0);
    vec4 vsPos4    = view_matrix * worldPos4;
    vec3 nVS       = normalize(normal_matrix * in_normal);

    vec4 clipPos   = mvp_matrix * worldPos4;
    gl_Position    = clipPos;

    // --- 1) RAW-Radius ---
    float r_raw = max(0.0, in_radius);

    // --- 2) Cut im RAW-Domain (vor Gamma & Scaling) ---
    if (max_radius_cut > 0.0 && r_raw > max_radius_cut) {
        gl_PointSize              = 0.0;                      // nichts rasterisieren
        gl_Position               = vec4(2e9, 2e9, 2e9, 1.0); // sicher außerhalb
        VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
        VertexOut.pass_world_pos   = worldPos4.xyz;
        VertexOut.pass_vs_pos      = vsPos4.xyz;
        VertexOut.pass_vs_normal   = nVS;
        VertexOut.pass_radius_ws   = 0.0;
        VertexOut.pass_screen_size = 0.0;
        return;
    }

    // --- 3) Shaping (Gamma) + Skalierung ins Weltmaß (Radius) ---
    float gamma  = (scale_radius_gamma > 0.0) ? scale_radius_gamma : 1.0;
    float r_ws_u = pow(r_raw, gamma) * scale_radius;   // ungeclampter WS-Radius

    // --- 4) Weltmaß-Clamp (Radius, nach Skalierung) ---
    float r_ws = clamp(r_ws_u, min_radius, max_radius);

    // --- 5) Screenspace-CLAMP auf Pixel-DURCHMESSER ---
    float w      = max(EPS, abs(clipPos.w));
    float d_px   = (2.0 * r_ws * scale_projection) / w;        // Pixel-Durchmesser
    float d_pxC  = clamp(d_px, min_screen_size, max_screen_size);

    if (d_pxC <= EPS) {
        gl_PointSize              = 0.0;
        VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
        VertexOut.pass_world_pos   = worldPos4.xyz;
        VertexOut.pass_vs_pos      = vsPos4.xyz;
        VertexOut.pass_vs_normal   = nVS;
        VertexOut.pass_radius_ws   = r_ws;
        VertexOut.pass_screen_size = 0.0;
        return;
    }

    gl_PointSize = d_pxC;

    // Outputs
    VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
    VertexOut.pass_world_pos   = worldPos4.xyz;
    VertexOut.pass_vs_pos      = vsPos4.xyz;
    VertexOut.pass_vs_normal   = nVS;
    VertexOut.pass_radius_ws   = r_ws;    // WS-Radius nach WS-Clamp
    VertexOut.pass_screen_size = d_pxC;   // Pixel-Durchmesser nach Pixel-Clamp
}