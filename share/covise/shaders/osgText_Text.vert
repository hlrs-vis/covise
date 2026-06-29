#if __VERSION__ >= 130
out vec2 texCoord;
out vec4 vertexColor;
#else
varying vec2 texCoord;
varying vec4 vertexColor;
#endif

void main(void)
{
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    texCoord = gl_MultiTexCoord0.xy;
    vertexColor = gl_Color;

#if !defined(GL_ES) && __VERSION__ < 140
    gl_ClipVertex = gl_ModelViewMatrix * gl_Vertex;
#endif
}
