/*
  Steffen Frey
  Fachpraktikum Graphik-Programmierung 2007
  Institut fuer Visualisierung und Interaktive Systeme
  Universitaet Stuttgart
 */

uniform sampler2D normalDepthMap;

uniform float width;
uniform float height;
varying vec3 normalPerVertex;
varying float depthInCamera;

uniform bool first;

void main()
{
  vec2 texCoord = vec2(gl_FragCoord.x/width, gl_FragCoord.y/height);
  float prevDepth = texture2D(normalDepthMap, texCoord).w;

  //peel away depth layers
  if(!first && depthInCamera <= prevDepth + 0.0005)
    discard;
  
  //normalize the per fragment normal
  vec3 normal = normalize(normalPerVertex);
  
  // Encode normals: [-1,1] to [0,1]
  //normal = (normal+1.0)*0.5;
  normal = abs(normal);
  
  // Output color and depth
  gl_FragColor = vec4(normal, depthInCamera);
  gl_FragDepth = depthInCamera;
}
