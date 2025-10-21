#version 420 core

out vec4 fragColor;

uniform vec4 in_color;

void main()
{
    fragColor = in_color;
}
