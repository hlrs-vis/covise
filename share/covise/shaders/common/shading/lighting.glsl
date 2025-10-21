// ../../common/shading/lighting.glsl

struct LightingParams {
    vec3  point_light_pos_vs;
    float point_light_intensity;
    float ambient_intensity;
    float specular_intensity;
    float shininess;
    float gamma;
    bool  use_tone_mapping;
};

const float PI = 3.14159265358979323846;

vec3 shade_blinn_phong(in vec3 vs_pos,
                       in vec3 vs_normal,
                       in vec3 vs_color,
                       in LightingParams lit)
{
    vec3 N = normalize(vs_normal);
    vec3 V = normalize(-vs_pos);
    vec3 L = normalize(lit.point_light_pos_vs - vs_pos);

    if (dot(N, V) < 0.0) N *= -1.0;

    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) {
        vec3 amb = lit.ambient_intensity * vs_color;
        vec3 mapped = lit.use_tone_mapping ? (amb / (amb + vec3(1.0))) : amb;
        return pow(mapped, vec3(1.0 / max(lit.gamma, 1e-6)));
    }

    vec3  H     = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);

    float n = (lit.shininess <= 1.0)
              ? exp2(mix(3.0, 10.0, clamp(lit.shininess, 0.0, 1.0)))
              : max(lit.shininess, 1.0);

    float specNorm = (n + 2.0) / (2.0 * PI);

    float k  = clamp(lit.specular_intensity, 0.0, 1.0);
    float f0 = mix(0.02, 0.10, 1.0 - exp(-k));

    vec3 kd = vec3(1.0 - f0);

    vec3 ambient = lit.ambient_intensity * vs_color;
    vec3 diffuse = lit.point_light_intensity * kd * vs_color * NdotL;

    float spec     = specNorm * pow(NdotH, n);
    vec3  specular = lit.point_light_intensity * vec3(f0) * spec * NdotL;

    vec3 shaded = ambient + diffuse + specular;

    if (lit.use_tone_mapping) {
        shaded = shaded / (shaded + vec3(1.0));
    }
    return pow(max(shaded, 0.0), vec3(1.0 / max(lit.gamma, 1e-6)));
}
