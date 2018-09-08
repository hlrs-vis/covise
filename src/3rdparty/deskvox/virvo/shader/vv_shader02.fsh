// Shader for 2-channel data sets and simple alpha blending

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
  OUT.y = texture2D(pixLUT, vec2(origColor.a, 0.0)).a;
  OUT.z = 0.0;
  OUT.w = max(OUT.x, OUT.y);

  gl_FragColor = OUT;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
