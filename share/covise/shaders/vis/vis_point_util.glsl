// vis_point_util.glsl
#ifndef VIS_POINT_UTIL_GLSL
#define VIS_POINT_UTIL_GLSL

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
float calc_world_radius(
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

float calc_diameter_px_iso(
    in vec4 clip,
    in float radius_ws,
    in float scale_projection,
    in float min_screen_size,
    in float max_screen_size)
{
    float w = max(UTIL_EPS, abs(clip.w));
    float diameter = (2.0 * radius_ws * scale_projection) / w;
    float clamped = clamp(diameter, min_screen_size, max_screen_size);
    return (clamped > UTIL_EPS) ? clamped : 0.0;
}

float calc_diameter_px_aniso(
    in vec4 clip,
    in float radius_ws,
    in vec4 Pcol0,
    in vec4 Pcol1,
    in float viewport_half_y,
    in float aniso_normalize,
    in float min_screen_size,
    in float max_screen_size)
{
    float invW = 1.0 / max(UTIL_EPS, abs(clip.w));
    vec2 a0 = (Pcol0.xy - clip.xy * (Pcol0.w * invW)) * invW;
    vec2 a1 = (Pcol1.xy - clip.xy * (Pcol1.w * invW)) * invW;
    float s2 = dot(a0, a0) + dot(a1, a1);
    float sRms = sqrt(max(0.0, 0.5 * s2));
    float radius_px = viewport_half_y * radius_ws * (sRms * aniso_normalize);
    float diameter = 2.0 * radius_px;
    float clamped = clamp(diameter, min_screen_size, max_screen_size);
    return (clamped > UTIL_EPS) ? clamped : 0.0;
}

#endif // VIS_POINT_UTIL_GLSL
