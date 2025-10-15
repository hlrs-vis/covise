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

    // 1. Calculate world-space radius using the utility function
    float r_ws = calculate_world_space_radius(
        in_radius,
        max_radius_cut,
        scale_radius_gamma,
        scale_radius,
        min_radius,
        max_radius
    );

    // 2. Calculate orthonormal basis using the utility function
    vec3 u, v;
    calculate_orthonormal_basis(in_normal, u, v);

    // 3. Transform to view-space
    vec3 vs_center = (model_view_matrix * vec4(in_position, 1.0)).xyz;
    vec3 vs_half_u_unscaled = (model_view_matrix * vec4(u * r_ws, 0.0)).xyz;
    vec3 vs_half_v_unscaled = (model_view_matrix * vec4(v * r_ws, 0.0)).xyz;

    // 4. Apply screen-space scaling and clamping (with Euclidean distance)
    float distance = length(vs_center);
    float w0 = max(EPS, distance);
    float d_px = (2.0 * r_ws * scale_projection) / w0;
    float d_pxC = clamp(d_px, min_screen_size, max_screen_size);
    float s = (d_px > EPS) ? (d_pxC / d_px) : 1.0;

    // 5. Output to geometry shader
    vs_out.vs_center = vs_center;
    vs_out.vs_half_u = vs_half_u_unscaled * s;
    vs_out.vs_half_v = vs_half_v_unscaled * s;

    gl_Position = projection_matrix * vec4(vs_center, 1.0);
}
