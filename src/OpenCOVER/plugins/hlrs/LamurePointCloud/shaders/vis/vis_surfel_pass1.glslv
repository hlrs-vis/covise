#version 420 core
INCLUDE vis_surfel_util.glsl

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
uniform vec2  viewport;
uniform bool  use_aniso;

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

    // Konsistente Welt-Radiusberechnung (inkl. CUT und CLAMP)
    float r_ws = map_raw_to_world_radius(
        in_radius,
        max_radius_cut,
        scale_radius_gamma,
        scale_radius,
        min_radius,
        max_radius);

    // Orthonormale Tangenten (Duff branchless)
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

    // nach View-Space (Halbachsen re-transformieren)
    vec3 vs_center = (mv * vec4(center_ws, 1.0)).xyz;
    vec3 vs_half_u = (mv * vec4(half_u_ws, 0.0)).xyz;
    vec3 vs_half_v = (mv * vec4(half_v_ws, 0.0)).xyz;

    // Output
    vs_out.vs_center = vs_center;
    vs_out.vs_half_u = vs_half_u;
    vs_out.vs_half_v = vs_half_v;

    gl_Position = projection_matrix * vec4(vs_center, 1.0);
}
