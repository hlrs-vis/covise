
#version 420 core

uniform sampler2D text;
uniform vec3 in_color;

in vec2 tex;
out vec4 color;

void main()
{
    float alpha = texture(text, tex).r;
    color = vec4(in_color, alpha);
}