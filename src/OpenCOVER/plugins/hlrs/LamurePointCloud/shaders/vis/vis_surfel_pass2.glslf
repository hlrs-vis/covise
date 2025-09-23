#version 420 core

uniform float depth_range;
uniform float flank_lift;

uniform sampler2D depth_texture;      // Prepass depth (GL_NEAREST)
uniform mat4      projection_matrix;  // P für NDC-Z-Vergleich

// --- Gaussian Lookup Table ---
const float gaussian[32] = float[](
  1.000000, 1.000000, 0.988235, 0.968627, 0.956862, 0.917647, 0.894117, 0.870588, 0.915686, 0.788235,
  0.749020, 0.690196, 0.654902, 0.619608, 0.552941, 0.513725, 0.490196, 0.458824, 0.392157, 0.356863,
  0.341176, 0.278431, 0.254902, 0.227451, 0.188235, 0.164706, 0.152941, 0.125490, 0.109804, 0.098039,
  0.074510, 0.062745
);

// --- In/Out Blocks ---
in GsOut {
    noperspective vec2 uv;   // -1..1
    flat vec3 vs_center;
    flat vec3 vs_half_u;
    flat vec3 vs_half_v;
    flat vec3 vs_normal;
    flat vec3 albedo_rgb;
} fs_in;

layout(location = 0) out vec4 accumulated_colors;       // rgb = Σ(color*w), a = Σw
layout(location = 1) out vec3 accumulated_normals;      // Σ(normal*w)
layout(location = 2) out vec3 accumulated_vs_positions; // Σ(pos_vs*w)

// --- Stability Constant ---
const float kEpsMin   = 1e-6; // Bleibt als Konstante

void main()
{
    vec2  uv = fs_in.uv;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    vec3 pos_vs = fs_in.vs_center + fs_in.vs_half_u * uv.x + fs_in.vs_half_v * uv.y;

    vec4  clip = projection_matrix * vec4(pos_vs, 1.0);
    float z01  = (clip.z / clip.w) * 0.5 + 0.5;
    float z_scene = texelFetch(depth_texture, ivec2(gl_FragCoord.xy), 0).r;

    // 1. Tiefentoleranz berechnen (mit neuem uniform)
    float eps_ndc = max(kEpsMin, fwidth(z01)) * depth_range;
    float dz = z01 - z_scene;
    if (dz > eps_ndc) discard;

    // 2. Gewichtung des Punkt-Profils berechnen
    float r = sqrt(r2);
    int idx = int(clamp(round(r * 31.0), 0.0, 31.0));
    float w_gauss = gaussian[idx];
    float w = mix(w_gauss, 1.0, flank_lift);

    // --- Final Output ---
    accumulated_colors       = vec4(fs_in.albedo_rgb * w, w);
    accumulated_normals      = fs_in.vs_normal * w;
    accumulated_vs_positions = pos_vs * w;
}
