// Authors of cg-version: 
//  Alexander Rice <acrice@cs.brown.edu>
//  Jurgen Schulze <schulze@cs.brown.edu>
//
// Converted by:
//  Stavros Delisavas <stavros.delisavas@uni-koeln.de>

uniform sampler2D pix2dtex;
uniform sampler2D pixLUT;

void main()
{
  vec4 origColor = texture2D(pix2dtex, gl_TexCoord[0].xy);
  vec4 OUT = texture2D(pixLUT, vec2(origColor.x, 0.0));
  gl_FragColor = OUT;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
