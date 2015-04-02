/*
  Steffen Frey
  Fachpraktikum Graphik-Programmierung 2007
  Institut fuer Visualisierung und Interaktive Systeme
  Universitaet Stuttgart
 */

uniform sampler2D tex;
uniform sampler2D normalDepthMap;

uniform float width;
uniform float height;

varying float depthInCamera;

void main( void )
{   
  vec2 texCoord = vec2(gl_FragCoord.x/width, gl_FragCoord.y/height);
  
  float prevDepth = texture2D(normalDepthMap, texCoord).w;
  
  if(depthInCamera > prevDepth + 0.0005)
    discard;
   
  gl_FragColor = vec4(gl_Color.xyz, 0.3);  
  gl_FragDepth = depthInCamera;    
}
