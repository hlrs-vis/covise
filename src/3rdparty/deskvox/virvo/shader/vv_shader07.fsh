// Shader for 3-channel data sets and opacity weighted alpha blending

// Authors of cg-version: 
//  Alexander Rice <acrice@cs.brown.edu>
//  Jurgen Schulze <schulze@cs.brown.edu>
//
// Converted by:
//   Stavros Delisavas <stavros.delisavas@uni-koeln.de>

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT;
uniform vec4 opWeights;

void main()
{
  vec4 origColor = texture3D(pix3dtex, gl_TexCoord[0].xyz);
  vec4 OUT;

  OUT.x = texture2D(pixLUT, vec2(origColor.x, 0.0)).x;
  OUT.y = texture2D(pixLUT, vec2(origColor.y, 0.0)).y;
  OUT.z = texture2D(pixLUT, vec2(origColor.z, 0.0)).z;

  float maxColor = max(OUT.x, max(OUT.y, OUT.z));

  if     (maxColor == OUT.x) OUT.w = OUT.x * opWeights.x;
  else if(maxColor == OUT.y) OUT.w = OUT.y * opWeights.y;
  else if(maxColor == OUT.z) OUT.w = OUT.z * opWeights.z;

  gl_FragColor = OUT;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
