void main( void )
{
  //gl_ClipVertex
  gl_TexCoord[1] = gl_MultiTexCoord1;
  //gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
  gl_Position = ftransform();
}
