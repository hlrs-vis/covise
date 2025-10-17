// vis_surfel.glslf (Fragment Shader)
#version 420 core

in FS_IN {
    vec3  pass_point_color;
    vec2  pass_uv_coords;
    vec3  pass_world_pos;
    float pass_radius_ws;   // effektiver Radius (WS) nach Pixel-Clamp
    float pass_screen_size; // effektiver Durchmesser in Pixeln
} fsIn;

layout(location = 0) out vec4 out_color;

void main() {
    // Kreisförmige Maske im Quad
    if (dot(fsIn.pass_uv_coords, fsIn.pass_uv_coords) > 1.0) {
        discard;
    }

    // Unbeleuchtete Ausgabe (Grundfarbe aus VS)
    out_color = vec4(fsIn.pass_point_color, 1.0);
}
