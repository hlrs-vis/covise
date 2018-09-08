// Authors of cg-version: 
//  Alexander Rice <acrice@cs.brown.edu>
//  Jurgen Schulze <schulze@cs.brown.edu>
//
// converted by:
// Stavros Delisavas <stavros.delisavas@uni-koeln.de>

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT;

void main()
{
    vec4 origColor = texture3D(pix3dtex, gl_TexCoord[0].xyz);
    gl_FragColor = texture2D(pixLUT, vec2(origColor.x, 0.0));
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
