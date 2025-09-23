#version 420 core

layout(binding = 0) uniform sampler2D in_color_texture;       // rgba: Σ(color*w), a: Σw
layout(binding = 1) uniform sampler2D in_normal_texture;      // rgb:  Σ(normal*w)
layout(binding = 2) uniform sampler2D in_vs_position_texture; // rgb:  Σ(pos_vs*w)

layout(location = 0) out vec4 out_color;

uniform vec3 background_color;
uniform bool lighting;

in VsOut {
    vec2 uv;  // 0..1
} fs_in;

INCLUDE ../common/shading/lighting.glsl

void main()
{
    vec2 t = fs_in.uv;

    vec4 accumulated_color  = texture(in_color_texture,       t);
    vec3 accumulated_normal = texture(in_normal_texture,      t).rgb;
    vec3 accumulated_pos_vs = texture(in_vs_position_texture, t).rgb;

    if (accumulated_color.a <= 0.0)
        discard; // kein Surfel-Beitrag

    float sum_w = accumulated_color.a; // Σw

    // Gemittelte Größen (gewichteter Mittelwert)
    vec3 albedo    = accumulated_color.rgb  / max(sum_w, 1e-8);
    vec3 normal_vs = normalize(accumulated_normal / max(sum_w, 1e-8));
    vec3 pos_vs    = accumulated_pos_vs     / max(sum_w, 1e-8);

    // Debug-Modi umgehen Beleuchtung
    vec3 shaded = (lighting) ? shade_blinn_phong(pos_vs, normal_vs, albedo) : albedo;

    // Voll deckende Ausgabe (keine Transparenz, kein Blend nötig)
    out_color = vec4(shaded, 1.0);
}