// Shader for four-channel data sets with alpha blending
//
// Authors of cg-version: 
//   Alexander Rice <acrice@cs.brown.edu>
//   Jurgen Schulze <schulze@cs.brown.edu>
//
// Converted by:
//   Stavros Delisavas <stavros.delisavas@uni-koeln.de>

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT;
uniform vec3 chan4color;
uniform vec4 opWeights;

void main()
{
  vec4 origColor = texture3D(pix3dtex, gl_TexCoord[0].xyz);
  vec4 intColor;
  vec4 OUT;

  intColor.x = texture2D(pixLUT, vec2(origColor.x, 0.0)).x;
  intColor.y = texture2D(pixLUT, vec2(origColor.y, 0.0)).y;
  intColor.z = texture2D(pixLUT, vec2(origColor.z, 0.0)).z;
  intColor.w = texture2D(pixLUT, vec2(origColor.w, 0.0)).w;

  OUT.x = max(intColor.x, (chan4color.x * intColor.w));
  OUT.y = max(intColor.y, (chan4color.y * intColor.w));
  OUT.z = max(intColor.z, (chan4color.z * intColor.w));

  float maxColor = max(intColor.x, max(intColor.y, max(intColor.z, intColor.w)));
  if      (maxColor == intColor.x) OUT.w = intColor.x * opWeights.x;
  else if (maxColor == intColor.y) OUT.w = intColor.y * opWeights.y;
  else if (maxColor == intColor.z) OUT.w = intColor.z * opWeights.z;
  else                             OUT.w = intColor.w * opWeights.w;

  gl_FragColor = OUT;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
