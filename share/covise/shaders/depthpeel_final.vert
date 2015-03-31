/*
  Steffen Frey
  Fachpraktikum Graphik-Programmierung 2007
  Institut fuer Visualisierung und Interaktive Systeme
  Universitaet Stuttgart
 */

void main( void )
{
  //gl_ClipVertex
  gl_TexCoord[0] = gl_MultiTexCoord0;
  gl_TexCoord[1] = gl_MultiTexCoord1;
  //gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
  gl_Position = ftransform();
}
