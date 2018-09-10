// Shader for 2D transfer functions.
//
// Author of cg-version: 
//  Jurgen Schulze <schulze@cs.brown.edu>
//
// Converted by:
//   Stavros Delisavas <stavros.delisavas@uni-koeln.de>

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT;

void main()
{
  vec4 origColor = texture3D(pix3dtex, gl_TexCoord[0].xyz);
  vec4 OUT = texture2D(pixLUT, vec2(origColor.x, origColor.w));
#ifdef LIGHTING
  OUT = light(pix3dtex, OUT, gl_TexCoord[0].xyz);
#endif
  gl_FragColor = OUT;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
