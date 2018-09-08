// Shader for four-channel data sets with alpha blending

// Authors of cg-version: 
//   Alexander Rice <acrice@cs.brown.edu>
//   Jurgen Schulze <schulze@cs.brown.edu>
//
// Converted by:
//   Stavros Delisavas <stavros.delisavas@uni-koeln.de>

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT; // used specifically for gamma correction
uniform vec3 chan4color;  // channel four color
uniform vec4 opWeights;   // opacity of four channels

void main()
{
  vec4 origColor = texture3D(pix3dtex, gl_TexCoord[0].xyz);  // color as stored in 3D texture
  vec4 intColor;    // intermediate color: after texture lookup (esp. gamma correction)
  vec4 OUT;         // result after color conversion

  intColor.x = texture2D(pixLUT, vec2(0.0, origColor.x)).x;
  intColor.y = texture2D(pixLUT, vec2(0.0, origColor.y)).y;
  intColor.z = texture2D(pixLUT, vec2(0.0, origColor.z)).z;
  intColor.w = texture2D(pixLUT, vec2(0.0, origColor.w)).w;

  OUT.x = max(intColor.x, chan4color.x * intColor.w);
  OUT.y = max(intColor.y, chan4color.y * intColor.w);
  OUT.z = max(intColor.z, chan4color.z * intColor.w);

  // maximum
  OUT.w = max(intColor.x * opWeights.x,
            max(intColor.y * opWeights.y,
              max(intColor.z * opWeights.z, intColor.w * opWeights.w)));

  gl_FragColor = OUT;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
