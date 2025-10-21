#version 420 core

in VertexData {
    vec4 pass_color;
} VertexIn;

out vec4 FragColor;

void main() {
    FragColor = VertexIn.pass_color;
}