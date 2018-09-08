// Shader for RGB data sets with alpha blending
//
// Author of cg-version: 
//  Martin Aumueller <aumueller@uni-koeln.de>
//
// Converted by:
//  Stavros Delisavas <stavros.delisavas@uni-koeln.de>

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT;

void main()
{
  vec4 c = texture3D(pix3dtex, gl_TexCoord[0].xyz);
  vec4 OUT;

  OUT.xyz = c.xyz;
  float t = (c.x + c.y + c.z)/3.0;
  OUT.w = texture2D(pixLUT, vec2(t, 0.0)).w;

  gl_FragColor = OUT;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
