// ../../common/shading/lighting.glsl

uniform vec3  point_light_pos_vs;
uniform float point_light_intensity;
uniform float ambient_intensity;
uniform float specular_intensity;
uniform float shininess;
uniform float gamma;
uniform bool  use_tone_mapping;


const float PI = 3.14159265358979323846;

vec3 shade_blinn_phong(in vec3 vs_pos, in vec3 vs_normal, in vec3 vs_color) 
{
    // Basis
    vec3 N = normalize(vs_normal);
    vec3 V = normalize(-vs_pos);
    vec3 L = normalize(point_light_pos_vs - vs_pos);

    // Beidseitig: zur Kamera orientieren
    if (dot(N, V) < 0.0) N *= -1.0;

    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0) {
        vec3 amb = ambient_intensity * vs_color;
        vec3 mapped = use_tone_mapping ? (amb / (amb + vec3(1.0))) : amb;
        return pow(mapped, vec3(1.0 / max(gamma, 1e-6)));
    }

    vec3  H     = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);

    // --- Kleine, aber wirkungsvolle Stabilisierungen ---
    // 1) Shininess: 0..1 als perzeptive Skala → Exponent 2^[3..10]; >1 = roher Exponent
    float n = (shininess <= 1.0)
              ? exp2(mix(3.0, 10.0, clamp(shininess, 0.0, 1.0)))
              : max(shininess, 1.0);

    // 2) Normierter Blinn-Phong (verhindert Überhellen bei großem n)
    float specNorm = (n + 2.0) / (2.0 * PI);

    // 3) Specular-Intensität sanft komprimieren und als F0 interpretieren
    float k  = clamp(specular_intensity, 0.0, 1.0);
    float f0 = mix(0.02, 0.10, 1.0 - exp(-k)); // 0.02..0.10, keine harten Peaks

    // Energieerhaltung light: reduziere Diffus um F0-Anteil
    vec3 kd = vec3(1.0 - f0);

    // Beleuchtung
    vec3 ambient = ambient_intensity * vs_color;
    vec3 diffuse = point_light_intensity * kd * vs_color * NdotL;

    float spec    = specNorm * pow(NdotH, n);
    vec3  specular= point_light_intensity * vec3(f0) * spec * NdotL;

    vec3 shaded = ambient + diffuse + specular;

    if (use_tone_mapping) {
        shaded = shaded / (shaded + vec3(1.0)); // Reinhard
    }
    return pow(max(shaded, 0.0), vec3(1.0 / max(gamma, 1e-6)));
}