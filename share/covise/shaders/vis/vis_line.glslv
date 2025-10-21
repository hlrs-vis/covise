#version 420 core

uniform mat4 mvp_matrix;
uniform vec4 in_color;

layout(location = 0) in vec3 in_position;

out VertexData {
    vec4 pass_color;
} VertexOut;

void main() {
    VertexOut.pass_color = in_color;
    gl_Position = mvp_matrix * vec4(in_position, 1.0);
}