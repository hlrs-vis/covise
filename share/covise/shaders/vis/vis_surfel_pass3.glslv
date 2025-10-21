#version 420 core

layout(location = 0) in vec3 in_position; // NDC-Quad [-1,1]

out VsOut { vec2 uv; } vs_out; // 0..1

void main() {
    gl_Position = vec4(in_position, 1.0);
    vs_out.uv = in_position.xy * 0.5 + 0.5;
}