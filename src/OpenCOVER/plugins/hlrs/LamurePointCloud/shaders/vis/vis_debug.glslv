#version 420 core

// Input is a simple quad (e.g., from a VBO)
layout(location = 0) in vec3 in_position;

// Pass-through texture coordinates to the fragment shader
out vec2 pos;

void main()
{
     // Create a screen-filling quad
     gl_Position = vec4(in_position, 1.0);

     // Pass the vertex position to use as texture coordinates
     pos = in_position.xy;
}