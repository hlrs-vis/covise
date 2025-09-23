#version 420 core

out vec4 FragColor;
in vec2 pos;

// Samplers for all G-Buffer textures
uniform sampler2D texture_depth;
uniform sampler2D texture_color; // Contains color in .rgb and weight in .a
uniform sampler2D texture_normal;
uniform sampler2D texture_position;

// Uniform to select what to debug
// 0: Depth, 1: Color, 2: Normal, 3: Position
uniform int debug_mode;

uniform float near_plane;
uniform float far_plane;

float linearize_depth(float depth) {
    float z = depth * 2.0 - 1.0; // zurück in NDC
    return (2.0 * near_plane * far_plane) / (far_plane + near_plane - z * (far_plane - near_plane));
}

void main()
{
    vec2 tex_coords = pos * 0.5 + 0.5;  // von [-1,1] → [0,1]
    float depth = texture(texture_depth, tex_coords).r;

    // Sanity check mode: Just draw a solid color
    if (debug_mode == -1) {
        FragColor = vec4(0.0, 0.0, 0.0, 1.0); // Black
        return;
    }

    if (debug_mode == 0) {
        // Lineare Tiefenvisualisierung
        FragColor = vec4(vec3(depth), 1.0);
        return;
    }
    if (debug_mode == 1) {
     // Akkumulierte Farb-+Gewichts-Daten aus Pass 2
     vec4 accum = texture(texture_color, tex_coords);
     // Wenn überhaupt Gewicht vorhanden, normalisiere Farb-RGB:
     if (accum.a > 0.0) {
          vec3 color = accum.rgb / accum.a;
          FragColor = vec4(color, 1.0);
     }
     else {
          FragColor = vec4(0.0, 0.0, 0.0, 1.0);
     }
     return;
    }
    if (debug_mode == 2) {
        // Normals (ausgesampelt und dekopmrimiert)
        vec4 cd = texture(texture_color, tex_coords);
        if (cd.a > 1e-3) {
            vec3 n = texture(texture_normal, tex_coords).xyz / cd.a;
            n = normalize(n);
            FragColor = vec4(n * 0.5 + 0.5, 1.0);
        }
        else {
            FragColor = vec4(0.0);
        }
        return;
    }
    if (debug_mode == 3) {
        // View-Space Position
        vec4 cd = texture(texture_color, tex_coords);
        if (cd.a > 1e-3) {
            vec3 p = texture(texture_position, tex_coords).xyz / cd.a;
            FragColor = vec4(abs(normalize(p)), 1.0);
        }
        else {
            FragColor = vec4(0.0);
        }
        return;
    }
    FragColor = vec4(1.0, 0.0, 1.0, 1.0);
}