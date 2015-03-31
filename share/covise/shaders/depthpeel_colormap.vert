/*
  Steffen Frey
  Fachpraktikum Graphik-Programmierung 2007
  Institut fuer Visualisierung und Interaktive Systeme
  Universitaet Stuttgart
 */

uniform float znear;
uniform float zfar;

//depth in camera space
varying float depthInCamera;

void main( void )
{
  vec4 position = gl_ModelViewMatrix * gl_Vertex;
  depthInCamera = (abs(position.z) - znear) / (zfar - znear);
  gl_Position = ftransform();
  gl_FrontColor = gl_Color;
}
