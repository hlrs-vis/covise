// vis_surfel_util.glsl
#ifndef VIS_SURFEL_UTIL_GLSL
#define VIS_SURFEL_UTIL_GLSL

// This header provides standalone helpers for surfel rendering. No uniforms are
// declared here; all data is passed as arguments for flexibility across stages.

const float UTIL_EPS = 1e-6;

// -------------------- Normal/Tangent utilities --------------------

// Branchless orthonormal basis from a unit normal (Duff, 2017)
// Input n should be normalized; falls back to +Z if near-zero
void onb_from_normal(in vec3 n_in, out vec3 u, out vec3 v) {
    vec3 n = normalize(length(n_in) > UTIL_EPS ? n_in : vec3(0,0,1));
    float s = sign(n.z);
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    u = normalize(vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x));
    v = normalize(vec3(b, s + n.y * n.y * a, -n.y));
}

// Original cross-ref method: pick a reference axis and build ONB via crosses
// Simple and fast, but can flip near axis-aligned configurations
void onb_ref_cross(in vec3 n_in, out vec3 u, out vec3 v) {
    vec3 n = normalize(length(n_in) > UTIL_EPS ? n_in : vec3(0,0,1));
    vec3 ref = (abs(n.x) > abs(n.y) && abs(n.x) > abs(n.z))
             ? vec3(0.0, 1.0, 0.0)
             : (abs(n.y) > abs(n.z) ? vec3(0.0, 0.0, 1.0)
                                      : vec3(1.0, 0.0, 0.0));
    u = normalize(cross(ref, n));
    v = normalize(cross(n, u));
}

// Frisvad ONB (branch on n.z). Continuous and robust, minimal ops.
// Ref: Frisvad, "Building an Orthonormal Basis, Revisited"
void onb_frisvad(in vec3 n_in, out vec3 u, out vec3 v) {
    vec3 n = normalize(length(n_in) > UTIL_EPS ? n_in : vec3(0,0,1));
    if (n.z < 0.0) {
        float a = 1.0/(1.0 - n.z);
        float b = n.x*n.y*a;
        u = vec3(1.0 - n.x*n.x*a, -b, n.x);
        v = vec3(b, n.y*n.y*a - 1.0, -n.y);
    } else {
        float a = 1.0/(1.0 + n.z);
        float b = -n.x*n.y*a;
        u = vec3(1.0 - n.x*n.x*a, b, -n.x);
        v = vec3(b, 1.0 - n.y*n.y*a, -n.y);
    }
    u = normalize(u);
    v = normalize(v);
}

// -------------------- Radius mapping --------------------

// Map raw radius to world-space radius with cut, gamma, scaling, clamp
float map_raw_to_world_radius(
    float raw_radius,
    float max_radius_cut,
    float scale_radius_gamma,
    float scale_radius,
    float min_radius_ws,
    float max_radius_ws)
{
    float r_raw = max(0.0, raw_radius);
    if ((max_radius_cut > 0.0) && (r_raw > max_radius_cut))
        return 0.0;
    float gamma = (scale_radius_gamma > 0.0) ? scale_radius_gamma : 1.0;
    float r_ws_unclamped = scale_radius * pow(r_raw, gamma);
    return clamp(r_ws_unclamped, min_radius_ws, max_radius_ws);
}

// -------------------- Projection helpers --------------------

// Convert clip-space to pixel coordinates given viewport = (width, height)
vec2 clip_to_pixel(in vec4 clip, in vec2 viewport) {
    float iw = (abs(clip.w) > UTIL_EPS) ? (1.0 / clip.w) : 0.0;
    vec2 ndc = clip.xy * iw;
    return 0.5 * (ndc + 1.0) * viewport;
}

// Compute NDC depth (0..1) from clip position
float clip_to_ndc_depth(in vec4 clip) {
    float iw = (abs(clip.w) > UTIL_EPS) ? (1.0 / clip.w) : 0.0;
    return (clip.z * iw) * 0.5 + 0.5;
}

// -------------------- Screen-space scaling --------------------

// Isotropic pixel-scaling: scales both axes uniformly to match clamped diameter
// Returns false if result should be discarded (too small)
// Depth-less isotropic scaling (fast path): standard surfel behavior
bool scale_isotropic_pixels(
    in vec3 center_ws,
    inout vec3 step_u_ws,
    inout vec3 step_v_ws,
    in mat4 mvp,
    in vec2 viewport,
    in float min_screen_size,
    in float max_screen_size,
    in float scale_projection, // Use the pre-calculated uniform
    out float out_pixel_diameter)
{
    vec4 Pc = mvp * vec4(center_ws, 1.0);
    float w0 = abs(Pc.w);
    if (w0 < UTIL_EPS) return false;

    float r_ws = length(step_u_ws);
    if (r_ws <= UTIL_EPS) return false;

    // Consistent calculation using the scale_projection uniform, like in the multi-pass shaders
    float d_px = (2.0 * r_ws * scale_projection) / w0;

    out_pixel_diameter = clamp(d_px, min_screen_size, max_screen_size);
    if (out_pixel_diameter <= UTIL_EPS) return false;

    float s = (d_px > UTIL_EPS) ? (out_pixel_diameter / d_px) : 1.0;
    step_u_ws *= s;
    step_v_ws *= s;
    return true;
}

// Anisotropic pixel-scaling (with depth): projects U/V to pixels and rescales separately
// Falls back to isotropic if degenerate or extremely anisotropic
bool scale_anisotropic_pixels_depth(
    in vec3 center_ws,
    inout vec3 step_u_ws,
    inout vec3 step_v_ws,
    in mat4 mvp,
    in vec2 viewport,
    in float min_screen_size,
    in float max_screen_size,
    out float out_pixel_diameter,
    out float out_ndc_depth)
{
    vec4 Pc = mvp * vec4(center_ws, 1.0);
    if (abs(Pc.w) <= UTIL_EPS) {
        return false;
    }
    vec4 Pu = mvp * vec4(center_ws + step_u_ws, 1.0);
    vec4 Pv = mvp * vec4(center_ws + step_v_ws, 1.0);
    out_ndc_depth = clip_to_ndc_depth(Pc);

    vec2 s0 = clip_to_pixel(Pc, viewport);
    vec2 su = clip_to_pixel(Pu, viewport);
    vec2 sv = clip_to_pixel(Pv, viewport);

    float ru = length(su - s0);
    float rv = length(sv - s0);

    float diameter_raw = 2.0f * max(ru, rv);
    if (diameter_raw <= UTIL_EPS)
        return false;

    float diameter_clamped = clamp(diameter_raw, min_screen_size, max_screen_size);
    float scale = diameter_clamped / diameter_raw;

    step_u_ws *= scale;
    step_v_ws *= scale;

    out_pixel_diameter = diameter_clamped;
    return diameter_clamped > UTIL_EPS;
}

// Anisotropic pixel-scaling (no depth): same as above, but does not report NDC depth
bool scale_anisotropic_pixels(
    in vec3 center_ws,
    inout vec3 step_u_ws,
    inout vec3 step_v_ws,
    in mat4 mvp,
    in vec2 viewport,
    in float min_screen_size,
    in float max_screen_size,
    out float out_pixel_diameter)
{
    vec4 Pc = mvp * vec4(center_ws, 1.0);
    if (abs(Pc.w) <= UTIL_EPS) {
        return false;
    }
    vec4 Pu = mvp * vec4(center_ws + step_u_ws, 1.0);
    vec4 Pv = mvp * vec4(center_ws + step_v_ws, 1.0);

    vec2 s0 = clip_to_pixel(Pc, viewport);
    vec2 su = clip_to_pixel(Pu, viewport);
    vec2 sv = clip_to_pixel(Pv, viewport);

    float ru = length(su - s0);
    float rv = length(sv - s0);

    float diameter_raw = 2.0f * max(ru, rv);
    if (diameter_raw <= UTIL_EPS)
        return false;

    float diameter_clamped = clamp(diameter_raw, min_screen_size, max_screen_size);
    float scale = diameter_clamped / diameter_raw;

    step_u_ws *= scale;
    step_v_ws *= scale;

    out_pixel_diameter = diameter_clamped;
    return diameter_clamped > UTIL_EPS;
}

// Isotropic scaling with depth reporting (compat variant for other shaders)
bool scale_isotropic_pixels_depth(
    in vec3 center_ws,
    inout vec3 step_u_ws,
    inout vec3 step_v_ws,
    in mat4 mvp,
    in vec2 viewport,
    in float min_screen_size,
    in float max_screen_size,
    in float scale_projection, // Use the pre-calculated uniform
    out float out_pixel_diameter,
    out float out_ndc_depth)
{
    vec4 Pc = mvp * vec4(center_ws, 1.0);
    out_ndc_depth = clip_to_ndc_depth(Pc);

    float w0 = abs(Pc.w);
    if (w0 < UTIL_EPS) return false;

    float r_ws = length(step_u_ws);
    if (r_ws <= UTIL_EPS) return false;

    // Consistent calculation using the scale_projection uniform
    float d_px = (2.0 * r_ws * scale_projection) / w0;

    out_pixel_diameter = clamp(d_px, min_screen_size, max_screen_size);
    if (out_pixel_diameter <= UTIL_EPS) return false;

    float s = (d_px > UTIL_EPS) ? (out_pixel_diameter / d_px) : 1.0;
    step_u_ws *= s;
    step_v_ws *= s;
    return true;
}

#endif // VIS_SURFEL_UTIL_GLSL
