#version 420 core

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

void main() {
    const float EPS = 1e-6;

    // Clip-/Gerätekoordinaten
    vec4 clip = mvp_matrix * vec4(in_position, 1.0);
    gl_Position = clip;

    // 1) RAW-Radius
    float r_raw = max(0.0, in_radius);

    // 2) CUT im RAW-Domain
    if (max_radius_cut > 0.0 && r_raw > max_radius_cut) {
        gl_PointSize    = 0.0;
        gl_Position     = vec4(2e9, 2e9, 2e9, 1.0); // sicher außerhalb
        VertexOut.color = vec3(0.0);
        return;
    }

    // 3) Gamma & Scale -> WS-Radius, danach WS-CLAMP (Radius)
    float gamma = (scale_radius_gamma > 0.0) ? scale_radius_gamma : 1.0;
    float r_ws = clamp(scale_radius * pow(r_raw, gamma), min_radius, max_radius);

    // 4) Screenspace-CLAMP auf Pixel-DURCHMESSER (gl_PointSize erwartet Durchmesser)
    float d_px  = (2.0 * r_ws * scale_projection) / max(EPS, abs(clip.w)); // Pixel-Durchmesser
    float d_pxC = clamp(d_px, min_screen_size, max_screen_size);

    if (d_pxC <= EPS) {
        gl_PointSize    = 0.0;
        VertexOut.color = vec3(0.0);
        return;
    }

    gl_PointSize    = d_pxC;
    VertexOut.color = vec3(in_r, in_g, in_b);
}