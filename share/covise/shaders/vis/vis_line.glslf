#version 420 core

in VertexData {
    vec4 pass_color;
} VertexIn;
 
layout(location = 0) out vec4 out_color;

void main() {
  out_color = vec4(VertexIn.pass_color);
}