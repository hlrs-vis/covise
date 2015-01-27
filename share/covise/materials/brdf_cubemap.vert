
varying vec3 N, L, H, R, T, B;              // Variables for calculating the brdf
varying vec3 BaseColor;                     // Color
varying float Transparency;

uniform vec3 LightPos;

void main()
{
   vec4 pos = gl_ModelViewMatrix * gl_Vertex;      // position of the vertex

   vec3 V = normalize(-pos.xyz);                   // viewing direction

   N = normalize(gl_NormalMatrix * gl_Normal);     // normal
   L = normalize(LightPos-pos.xyz);                // light direction
   H = normalize(L + V);                           // halfvector
   R = reflect(pos.xyz, N);

   T = normalize(cross(N,vec3(1.0,0.0,0.0)));
   B = normalize(cross(N,T));

   Transparency = gl_Color.a;
   BaseColor = gl_Color.rgb;

   gl_Position = ftransform();
   gl_ClipVertex = pos;
   gl_TexCoord[0] = gl_MultiTexCoord0;
}
