/*
  Steffen Frey
  Fachpraktikum Graphik-Programmierung 2007
  Institut fuer Visualisierung und Interaktive Systeme
  Universitaet Stuttgart
 */

uniform float znear;
uniform float zfar;

//normal per vertx
varying vec3 normalPerVertex;

//depth in camera space
varying float depthInCamera;

vec3 fnormal(void)
{
  vec3 normal = gl_NormalMatrix * gl_Normal;
  normal = normalize(normal);
  return normal;
}

void main()
{
  gl_TexCoord[0] = gl_MultiTexCoord0;
  normalPerVertex = fnormal();
  vec4 position = gl_ModelViewMatrix * gl_Vertex;
  depthInCamera = (abs(position.z) - znear) / (zfar - znear);
  gl_Position = ftransform();
}
