/*
  Steffen Frey
  Fachpraktikum Graphik-Programmierung 2007
  Institut fuer Visualisierung und Interaktive Systeme
  Universitaet Stuttgart
 */

//these are for debugging purposes only
uniform sampler2D normalDepthMap0;
uniform sampler2D normalDepthMap1;

uniform sampler2D edgeMap;
uniform sampler2D colorMap;
uniform sampler2D noiseMap;

uniform bool sketchy;
uniform bool colored;
uniform bool edgy;

uniform float sketchiness;

void main( void )
{ 
  vec4 color;
  vec4 edge;
  //sketchy
  if(sketchy)
    {
      vec2 off = vec2(texture2D(noiseMap, gl_TexCoord[1].st).x, 
		      texture2D(noiseMap, vec2(1. - gl_TexCoord[1].s, 1. - gl_TexCoord[1].t)).x);
      off = 2. * off - 1.;
      vec4 a = sketchiness * vec4(0.007, 0.005, 0.006, 0.004);
      vec2 stEdge = gl_TexCoord[1].st + vec2(a[0]*off[0] + a[1]*off[1], a[2]*off[0] + a[3] * off[1]);
      
      vec4 b = sketchiness * vec4(0.014, 0.012, 0.016, 0.01);
      vec2 stColor = gl_TexCoord[1].st + vec2(b[0]*off[0] + b[1]*off[1], b[2]*off[0] + b[3] * off[1]);
  
      float borderWidth = 0.005;
      
      color = texture2D(colorMap, vec2(clamp(stColor.s, borderWidth, 1.-borderWidth) ,
                                       clamp(stColor.t, borderWidth, 1.-borderWidth)));
      edge = texture2D(edgeMap, vec2(clamp(stEdge.s, borderWidth, 1.-borderWidth) ,
                                      clamp(stEdge.t, borderWidth, 1.-borderWidth)));
      
    }
  else
  {
      color = texture2D(colorMap, gl_TexCoord[1].st);
      edge = texture2D(edgeMap, gl_TexCoord[1].st);
    }
  
  vec4 composed = vec4(max(color.x - (1.0 - edge.x), 0.0),
		       max(color.y - (1.0 - edge.y), 0.0),
		       max(color.z - (1.0 - edge.z), 0.0),
		       1.0);
  
  
  if(colored && edgy)
  {
      gl_FragColor = composed;
  }
  else if(edgy)
  {
      gl_FragColor = vec4(edge.xyz, 1.0);
  }
  else if(colored)
  {
      gl_FragColor = vec4(color.xyz, 1.0);
  }
  else
  {
      gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0);
  }
  
}
