#version 420 core

// Layout 0: Dummy-Vertex (wird nicht weiter genutzt)
layout(location = 0) in vec3 dummy;

// Layout 1: Instanzattribut für die minimale Koordinate
layout(location = 1) in vec3 instanceMin;

// Layout 2: Instanzattribut für die maximale Koordinate
layout(location = 2) in vec3 instanceMax;

// Übergabe an den Geometry Shader
out vec3 vs_min;
out vec3 vs_max;

void main()
{
    vs_min = instanceMin;
    vs_max = instanceMax;
    // gl_Position wird hier nicht benutzt, da der Geometry Shader die echten Positionen berechnet.
    gl_Position = vec4(0.0);
}
