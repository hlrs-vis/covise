// Shader for 4 channels with simple alpha

// Authors of cg-version: 
//   Alexander Rice <acrice@cs.brown.edu>
//   Jurgen Schulze <schulze@cs.brown.edu>
//
// Converted by:
//   Stavros Delisavas <stavros.delisavas@uni-koeln.de>

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT;
uniform vec3 chan4color; // channel for weights

void main()
{
  vec4 origColor = texture3D(pix3dtex, gl_TexCoord[0].xyz); // color as stored in 3D texture
  vec4 intColor;    // intermediate color
  vec4 OUT;         // result after color conversion

  intColor.x = texture2D(pixLUT, vec2(origColor.x, 0.0)).x;
  intColor.y = texture2D(pixLUT, vec2(origColor.y, 0.0)).y;
  intColor.z = texture2D(pixLUT, vec2(origColor.z, 0.0)).z;
  intColor.w = texture2D(pixLUT, vec2(origColor.w, 0.0)).w;

  OUT.x = max(intColor.x, chan4color.x * intColor.w);
  OUT.y = max(intColor.y, chan4color.y * intColor.w);
  OUT.z = max(intColor.z, chan4color.z * intColor.w);

  OUT.w = max(intColor.x, max(intColor.y, max(intColor.z, intColor.w)));
  gl_FragColor = OUT;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
