// Shader for 3-channel data sets and simple alpha blending

// Authors: 
//  Alexander Rice <acrice@cs.brown.edu>
//  Jurgen Schulze <schulze@cs.brown.edu>
//
// Converted by:
//   Stavros Delisavas <stavros.delisavas@uni-koeln.de>

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT;

void main()
{
  vec4 origColor = texture3D(pix3dtex, gl_TexCoord[0].xyz);
  vec4 OUT;

  OUT.x = texture2D(pixLUT, vec2(origColor.x, 0.0)).x;
  OUT.y = texture2D(pixLUT, vec2(origColor.y, 0.0)).y;
  OUT.z = texture2D(pixLUT, vec2(origColor.z, 0.0)).z;
  OUT.w = max(OUT.x, max(OUT.y, OUT.z));
  gl_FragColor = OUT;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
