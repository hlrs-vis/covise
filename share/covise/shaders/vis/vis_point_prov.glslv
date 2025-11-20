// vis_point_prov.glslv (Vertex Shader)
#version 420 core

INCLUDE vis_clip_util.glsl

uniform int clip_plane_count;
uniform vec4 clip_planes[LAMURE_MAX_CLIP_PLANES];
uniform mat4 model_matrix;

// --- Inputs ---
layout(location = 0) in vec3  in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 4) in float empty;
layout(location = 5) in float in_radius;   // RAW-RADIUS
layout(location = 6) in vec3  in_normal;

// Provenance-Inputs
layout(location = 7)  in float in_prov1;
layout(location = 8)  in float in_prov2;
layout(location = 9)  in float in_prov3;
layout(location = 10) in float in_prov4;
layout(location = 11) in float in_prov5;
layout(location = 12) in float in_prov6;

// --- Uniforms ---
uniform mat4  mvp_matrix;
uniform float max_radius;         // WS-CLAMP (Radius)
uniform float min_radius;         // WS-CLAMP (Radius)
uniform float min_screen_size;    // Screenspace-CLAMP (Pixel-Durchmesser)
uniform float max_screen_size;    // Screenspace-CLAMP (Pixel-Durchmesser)
uniform float scale_radius;       // RAW-Radius -> WS-Radius
uniform float scale_projection;   // Weltmaß -> Pixel (vor /w)

// --- Outputs an Fragment Shader ---
out VertexData {
    vec3  pass_point_color;   // entspricht in_r/g/b
    vec3  pass_world_pos;     // Position in Welt
    vec3  pass_normal_ws;     // Normale im World-Space
    float pass_radius_ws;     // WS-Radius nach Welt-Clamp
    float pass_screen_size;   // Pixel-Durchmesser nach Pixel-Clamp (= gl_PointSize)
    float pass_prov1;         // Provenance-Kanäle
    float pass_prov2;
    float pass_prov3;
    float pass_prov4;
    float pass_prov5;
    float pass_prov6;
} VertexOut;

void main() {
    const float EPS = 1e-6;

    if (clip_plane_count > 0)
    {
        vec4 clipWorldPos = model_matrix * vec4(in_position, 1.0);
        lamure_apply_clip_planes(clipWorldPos, clip_plane_count, clip_planes);
    }
    else
    {
        lamure_apply_clip_planes(vec4(0.0), 0, clip_planes);
    }

    // Clip-/Gerätekoordinaten
    vec4 clipPos = mvp_matrix * vec4(in_position, 1.0);
    gl_Position  = clipPos;

    // 1) RAW-Radius (hier ohne Gamma/Cut, da nicht als Uniform vorhanden)
    float r_raw = max(0.0, in_radius);

    // 2) RAW -> WS-Radius + WORLD-CLAMP (Radien!)
    float r_ws_u = scale_radius * r_raw;                // ungeclampter WS-Radius
    float r_ws   = clamp(r_ws_u, min_radius, max_radius);

    // 3) Screenspace-CLAMP auf Pixel-DURCHMESSER (gl_PointSize erwartet Durchmesser)
    float w     = max(EPS, abs(clipPos.w));
    float d_px  = (2.0 * r_ws * scale_projection) / w;  // Pixel-Durchmesser
    float d_pxC = clamp(d_px, min_screen_size, max_screen_size);

    if (d_pxC <= EPS) {
        gl_PointSize = 0.0;
        // Outputs (konstant/neutral befüllen)
        VertexOut.pass_point_color = vec3(0.0);
        VertexOut.pass_world_pos   = in_position;
        VertexOut.pass_normal_ws   = normalize((length(in_normal) > EPS) ? in_normal : vec3(0,0,1));
        VertexOut.pass_radius_ws   = r_ws;
        VertexOut.pass_screen_size = 0.0;
        VertexOut.pass_prov1 = in_prov1;
        VertexOut.pass_prov2 = in_prov2;
        VertexOut.pass_prov3 = in_prov3;
        VertexOut.pass_prov4 = in_prov4;
        VertexOut.pass_prov5 = in_prov5;
        VertexOut.pass_prov6 = in_prov6;
        return;
    }

    gl_PointSize = d_pxC;

    // Outputs
    VertexOut.pass_point_color = vec3(in_r, in_g, in_b);
    VertexOut.pass_world_pos   = in_position;
    VertexOut.pass_normal_ws   = normalize((length(in_normal) > EPS) ? in_normal : vec3(0,0,1));
    VertexOut.pass_radius_ws   = r_ws;    // WS-Radius nach WS-CLAMP
    VertexOut.pass_screen_size = d_pxC;   // Pixel-Durchmesser nach Pixel-CLAMP
    VertexOut.pass_prov1 = in_prov1;
    VertexOut.pass_prov2 = in_prov2;
    VertexOut.pass_prov3 = in_prov3;
    VertexOut.pass_prov4 = in_prov4;
    VertexOut.pass_prov5 = in_prov5;
    VertexOut.pass_prov6 = in_prov6;
}
