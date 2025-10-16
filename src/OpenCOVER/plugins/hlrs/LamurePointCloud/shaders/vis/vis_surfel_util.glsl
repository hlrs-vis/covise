#ifndef VIS_SURFEL_UTIL_GLSL
#define VIS_SURFEL_UTIL_GLSL

const float UTIL_EPS = 1e-6;

// --- Vertex Shader Utilities ---

// Calculates a world-space radius from a raw input radius, applying cutting, gamma, scaling, and clamping.
float calculate_world_space_radius(
    float raw_radius,
    float max_radius_cut,
    float scale_radius_gamma,
    float scale_radius,
    float min_radius_ws,
    float max_radius_ws
) {
    float r_raw = max(0.0, raw_radius);
    bool cut = (max_radius_cut > 0.0) && (r_raw > max_radius_cut);
    if (cut) {
        return 0.0;
    }

    float gamma = (scale_radius_gamma > 0.0) ? scale_radius_gamma : 1.0;
    float r_ws_unclamped = scale_radius * pow(r_raw, gamma);
    return clamp(r_ws_unclamped, min_radius_ws, max_radius_ws);
}

// Calculates an orthonormal basis (tangent, bitangent) from a given normal vector.
void calculate_orthonormal_basis(vec3 normal, out vec3 tangent, out vec3 bitangent) {
    vec3 n = normalize(length(normal) > UTIL_EPS ? normal : vec3(0,0,1));
    vec3 ref = (abs(n.x)>abs(n.y) && abs(n.x)>abs(n.z)) ? vec3(0,1,0)
              : (abs(n.y)>abs(n.z) ? vec3(0,0,1) : vec3(1,0,0));
    tangent = normalize(cross(ref, n));
    bitangent = normalize(cross(n, tangent));
}


// --- Geometry Shader Utilities ---

// Calculates the screen-space size of a surfel, clamps it,
// and rescales the world-space axes vectors accordingly.
// Returns true if the surfel is visible, false otherwise.
bool scale_surfel_for_screen(
    vec3 center_ws,
    mat4 model_view_matrix,
    float scale_projection,
    float min_screen_size,
    float max_screen_size,
    inout vec3 step_u_ws,
    inout vec3 step_v_ws,
    out float out_pixel_diameter,
    out float out_world_radius
) {
    // 1. Calculate Euclidean distance
    vec4 center_eye = model_view_matrix * vec4(center_ws, 1.0);
    float distance = length(center_eye.xyz);
    float w0 = max(UTIL_EPS, distance);

    // 2. Calculate initial projected pixel diameter
    float r_ws = length(step_u_ws);
    if (r_ws <= UTIL_EPS) {
        return false; // Not visible
    }
    float d_ws = 2.0 * r_ws;
    float d_px = (d_ws * scale_projection) / w0;

    // 3. Clamp pixel diameter
    out_pixel_diameter = clamp(d_px, min_screen_size, max_screen_size);
    if (out_pixel_diameter <= UTIL_EPS) {
        return false; // Not visible
    }

    // 4. Calculate and apply scaling
    float scale_factor = (d_px > UTIL_EPS) ? (out_pixel_diameter / d_px) : 1.0;
    step_u_ws *= scale_factor;
    step_v_ws *= scale_factor;

    // 5. Set final world radius
    out_world_radius = r_ws * scale_factor;

    return true; // Visible
}

#endif // VIS_SURFEL_UTIL_GLSL
