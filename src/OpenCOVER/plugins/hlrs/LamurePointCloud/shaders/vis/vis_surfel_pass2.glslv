#version 420 core

uniform float max_radius;
uniform float min_radius;
uniform float scale_radius;

uniform mat3  normal_matrix;
uniform mat4  model_view_matrix;
uniform mat4  projection_matrix;

uniform vec2  viewport;
uniform bool  coloring;
uniform bool  show_normals;
uniform bool  show_accuracy;
uniform bool  show_radius_deviation;
uniform bool  show_output_sensitivity;
uniform float accuracy;
uniform float average_radius;

uniform float min_screen_size;
uniform float max_screen_size;
uniform float scale_projection;
uniform float scale_radius_gamma; 
uniform float max_radius_cut;

layout(location = 0) in vec3  in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 5) in float in_radius;   // Roh-Wert, wird zu DURCHMESSER skaliert
layout(location = 6) in vec3  in_normal;

INCLUDE ../common/heatmapping/wavelength_to_rainbow.glsl
INCLUDE ../common/heatmapping/colormap.glsl

out VsOut {
    vec3 vs_center;
    vec3 vs_half_u;
    vec3 vs_half_v;
    vec3 vs_normal;
    vec3 albedo_rgb;
} vs_out;

float compute_screen_size_px(in vec3 vs_center,
                             in vec3 vs_half_u,
                             in vec3 vs_half_v,
                             in vec2 viewport)
{
    const float EPS = 1e-6;

    // Projektionsraum-Koordinaten (Clip Space)
    vec4 c0 = projection_matrix * vec4(vs_center,                1.0);
    vec4 cu = projection_matrix * vec4(vs_center + vs_half_u,    1.0);
    vec4 cv = projection_matrix * vec4(vs_center + vs_half_v,    1.0);

    // NDC (perspective divide) – individuell pro Punkt
    float iw0 = (abs(c0.w) > EPS) ? 1.0 / c0.w : 0.0;
    float iwu = (abs(cu.w) > EPS) ? 1.0 / cu.w : 0.0;
    float iwv = (abs(cv.w) > EPS) ? 1.0 / cv.w : 0.0;

    vec2 ndc0 = c0.xy * iw0;
    vec2 ndcu = cu.xy * iwu;
    vec2 ndcv = cv.xy * iwv;

    // NDC → Pixel (Viewport enthält [width, height] in Pixeln)
    vec2 s0 = 0.5 * (ndc0 + 1.0) * viewport;
    vec2 su = 0.5 * (ndcu + 1.0) * viewport;
    vec2 sv = 0.5 * (ndcv + 1.0) * viewport;

    // Pixel-Radien entlang U/V und daraus der Durchmesser
    float ru = length(su - s0);
    float rv = length(sv - s0);
    float diameter_px = 2.0 * max(ru, rv);

    // Falls gewünscht, hier zusätzlich clampen:
    // diameter_px = clamp(diameter_px, min_screen_size, max_screen_size);

    return diameter_px;
}

vec3 get_output_sensitivity_color(float screen_size) {
    float min_screen_size = 0.0;
    float max_screen_size = 10.0;
    float val = clamp(screen_size, min_screen_size, max_screen_size);
    return data_value_to_rainbow(val, min_screen_size, max_screen_size);
}


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

    vec3 n_ws = normalize((length(in_normal) > EPS) ? in_normal : vec3(0,0,1));
    vec3 ref  = (abs(n_ws.x)>abs(n_ws.y) && abs(n_ws.x)>abs(n_ws.z)) ? vec3(0,1,0)
               : (abs(n_ws.y)>abs(n_ws.z) ? vec3(0,0,1) : vec3(1,0,0));
    vec3 ms_u = normalize(cross(ref, n_ws));
    vec3 ms_v = normalize(cross(n_ws, ms_u));

    vec3 vs_center = (model_view_matrix * vec4(in_position, 1.0)).xyz;
    vec3 vs_half_u = (model_view_matrix * vec4(ms_u * r_ws, 0.0)).xyz; // Radius!
    vec3 vs_half_v = (model_view_matrix * vec4(ms_v * r_ws, 0.0)).xyz;

    // --- Pixel-RADIUS clamp + isotrope Skalierung ---
    float w0   = max(EPS, abs((projection_matrix * vec4(vs_center, 1.0)).w));
    float Rpx  = (r_ws * scale_projection) / w0;                 // Pixel-RADIUS
    float RpxC = clamp(Rpx, min_screen_size, max_screen_size);
    float s    = (Rpx > EPS) ? (RpxC / Rpx) : 1.0;

    vs_half_u *= s;
    vs_half_v *= s;

    vec3 base_color = vec3(in_r, in_g, in_b);
    vec3 vs_normal  = normalize(normal_matrix * in_normal);
    if (length(vs_normal) < EPS) vs_normal = vec3(0,0,1);

    if (coloring) {
        if (show_normals) {
            vec3 n_vis = vs_normal; if (n_vis.z < 0.0) n_vis *= -1.0;
            base_color = n_vis * 0.5 + 0.5;
        }
        if (show_radius_deviation) {
            float safe_avg = max(1e-8, average_radius);
            float rel      = r_ws / safe_avg;
            float max_fac  = 2.0;
            base_color     = vec3(min(max_fac, rel) / max_fac);
        }
        if (show_accuracy) base_color += vec3(accuracy, 0.0, 0.0);
        if (show_output_sensitivity) {
            float screen_size = compute_screen_size_px(vs_center, vs_half_u, vs_half_v, viewport);
            base_color = get_output_sensitivity_color(screen_size);
        }
    }
    
    vs_out.vs_center  = vs_center;
    vs_out.vs_half_u  = vs_half_u;
    vs_out.vs_half_v  = vs_half_v;
    vs_out.vs_normal  = vs_normal;
    vs_out.albedo_rgb = base_color;

    gl_Position = projection_matrix * vec4(vs_center, 1.0);
}
