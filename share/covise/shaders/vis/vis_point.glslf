#version 420 core

in VertexData {
    vec3 color;
} Fragment;

out vec4 out_color;

void main() {
    out_color = vec4(Fragment.color, 1.0);
}