
#version 420 core

uniform mat4 projection;

layout(location = 0) in vec4 vertex;
out vec2 tex;


void main()
{
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    tex = vertex.zw;
}