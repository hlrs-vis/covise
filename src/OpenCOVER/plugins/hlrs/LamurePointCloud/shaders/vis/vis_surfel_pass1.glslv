#version 420 core

uniform float max_radius;          // WS-CLAMP (Radius)
uniform float min_radius;          // WS-CLAMP (Radius)
uniform float scale_radius;        // RAW-Radius -> WS-Radius
uniform float scale_radius_gamma;  // Gamma auf RAW-Radius
uniform float max_radius_cut;      // CUT im RAW-RADIUS (vor Skalierung!)

uniform mat4  model_view_matrix;
uniform mat4  projection_matrix;
uniform mat3  normal_matrix;

uniform float min_screen_size;
uniform float max_screen_size;
uniform float scale_projection;

layout(location = 0) in vec3  in_position;
layout(location = 5) in float in_radius;
layout(location = 6) in vec3  in_normal;

out VsOut {
    vec3 vs_center;
    vec3 vs_half_u;
    vec3 vs_half_v;
} vs_out;

void main() {
    const float EPS = 1e-6;

    // --- RAW-Radius & Cut im RAW-Domain ---
    float r_raw = max(0.0, in_radius);
    bool cut = (max_radius_cut > 0.0) && (r_raw > max_radius_cut);

    // --- WS-Radius mit Gamma/Scale, danach CLAMP im WS (Radius) ---
    float r_ws = 0.0;
    if (!cut) {
        float gamma = (scale_radius_gamma > 0.0) ? scale_radius_gamma : 1.0;
        float r_ws_unclamped = scale_radius * pow(r_raw, gamma); // WS-Radius
        r_ws = clamp(r_ws_unclamped, min_radius, max_radius);    // CLAMP im WS (Radius)
    }

    // Orthonormale Tangenten
    vec3 n_ws = normalize((length(in_normal) > EPS) ? in_normal : vec3(0,0,1));
    vec3 ref  = (abs(n_ws.x)>abs(n_ws.y) && abs(n_ws.x)>abs(n_ws.z)) ? vec3(0,1,0)
               : (abs(n_ws.y)>abs(n_ws.z) ? vec3(0,0,1) : vec3(1,0,0));
    vec3 ms_u = normalize(cross(ref, n_ws));
    vec3 ms_v = normalize(cross(n_ws, ms_u));

    // nach View-Space
    vec3 vs_center = (model_view_matrix * vec4(in_position, 1.0)).xyz;
    vec3 vs_half_u = (model_view_matrix * vec4(ms_u * r_ws, 0.0)).xyz; // Radius!
    vec3 vs_half_v = (model_view_matrix * vec4(ms_v * r_ws, 0.0)).xyz;

    // --- Screenspace-CLAMP auf Pixel-DURCHMESSER + isotrope Skalierung ---
    float w0   = max(EPS, abs((projection_matrix * vec4(vs_center, 1.0)).w));
    float d_px = (2.0 * r_ws * scale_projection) / w0;          // Pixel-Durchmesser
    float d_pxC = clamp(d_px, min_screen_size, max_screen_size); // Clamp (Durchmesser)
    float s    = (d_px > EPS) ? (d_pxC / d_px) : 1.0;

    vs_half_u *= s;  // Radien isotrop skalieren
    vs_half_v *= s;

    // Output
    vs_out.vs_center = vs_center;
    vs_out.vs_half_u = vs_half_u;
    vs_out.vs_half_v = vs_half_v;

    gl_Position = projection_matrix * vec4(vs_center, 1.0);
}
