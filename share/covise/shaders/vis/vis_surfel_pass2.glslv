#version 420 core
INCLUDE vis_surfel_util.glsl

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
uniform bool  use_aniso;

layout(location = 0) in vec3  in_position;
layout(location = 1) in float in_r;
layout(location = 2) in float in_g;
layout(location = 3) in float in_b;
layout(location = 5) in float in_radius;   // Roh-RADIUS aus Vorverarbeitung
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

    float r_ws = map_raw_to_world_radius(
        in_radius,
        max_radius_cut,
        scale_radius_gamma,
        scale_radius,
        min_radius,
        max_radius);

    vec3 n_ws = normalize((length(in_normal) > EPS) ? in_normal : vec3(0,0,1));
    vec3 ms_u, ms_v;
    onb_from_normal(n_ws, ms_u, ms_v);

    vec3 center_ws = in_position;
    vec3 half_u_ws = ms_u * r_ws;
    vec3 half_v_ws = ms_v * r_ws;

    mat4 mv  = model_view_matrix;
    mat4 mvp = projection_matrix * mv;

    float pixel_diameter = 0.0;
    bool ok = false;
    if (use_aniso) {
        ok = scale_anisotropic_pixels(center_ws, half_u_ws, half_v_ws,
                                      mvp, viewport,
                                      min_screen_size, max_screen_size,
                                      pixel_diameter);
    } else {
        ok = scale_isotropic_pixels(center_ws, half_u_ws, half_v_ws,
                                    mvp, viewport,
                                    min_screen_size, max_screen_size,
                                    scale_projection,
                                    pixel_diameter);
    }
    if (!ok) {
        half_u_ws = vec3(0.0);
        half_v_ws = vec3(0.0);
    }

    vec3 vs_center = (mv * vec4(center_ws, 1.0)).xyz;
    vec3 vs_half_u = (mv * vec4(half_u_ws, 0.0)).xyz;
    vec3 vs_half_v = (mv * vec4(half_v_ws, 0.0)).xyz;

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
