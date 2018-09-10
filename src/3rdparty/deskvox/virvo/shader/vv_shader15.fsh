// Shader for 4 channels where alpha of 4th TF controls opacity of RGB in first 3 channels

// Author: Martin Aumueller <aumueller@hlrs.de>

uniform int channels;
uniform int preintegration;
uniform int lighting;

uniform sampler3D pix3dtex;

uniform sampler2D pixLUT;

void main()
{
  bool preint = preintegration==0 ? false : true;
  vec4 data = texture3D(pix3dtex, gl_TexCoord[0].xyz); // data from texture for up to 4 channels
  vec3 tc = gl_TexCoord[0].xyz;
#ifdef PREINTEGRATION
  vec4 data1 = texture3D(pix3dtex, gl_TexCoord[1].xyz);
  tc += gl_TexCoord[1].xyz;
  tc *= 0.5;
#else
  vec4 data1 = vec4(0., 0., 0., 0.);
#endif

  vec4 c = data.rgba;
#ifdef LIGHTING
  c = light(pix3dtex, c, tc);
#endif

  vec4 ca = classify(pixLUT, data.a, data1.a, preint);
  c.a = ca.a;

  gl_FragColor.a = c.a;
  gl_FragColor.rgb = c.rgb;

}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
