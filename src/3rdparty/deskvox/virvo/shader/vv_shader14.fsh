// Shader for up to 4 channels with independent TFs

// Author: Martin Aumueller <aumueller@hlrs.de>

uniform int channels;
uniform int preintegration;
uniform int lighting;
uniform float channelWeights[NUM_CHANNELS];

uniform sampler3D pix3dtex;

uniform sampler2D pixLUT0;
uniform sampler2D pixLUT1;
uniform sampler2D pixLUT2;
uniform sampler2D pixLUT3;

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

  vec4 c[NUM_CHANNELS];
  c[0] = classify(pixLUT0, data.x, data1.x, preint);
#ifdef LIGHTING
  c[0] = light(pix3dtex, c[0], tc);
#endif
#if NUM_CHANNELS > 1
#ifdef CHANNEL_WEIGHTS
  c[0].a *= channelWeights[0];
#endif
  float maxAlpha = c[0].a;
#if NUM_CHANNELS == 2
  c[1] = classify(pixLUT1, data.w, data1.w, preint);
#ifdef LIGHTING
  c[1] = light(pix3dtex, c[1], tc);
#endif
#ifdef CHANNEL_WEIGHTS
  c[1].a *= channelWeights[1];
#endif
  maxAlpha = max(maxAlpha, c[1].a);
#elif NUM_CHANNELS >= 3
  c[1] = classify(pixLUT1, data.y, data1.y, preint);
#ifdef LIGHTING
  c[1] = light(pix3dtex, c[1], tc);
#endif
#ifdef CHANNEL_WEIGHTS
  c[1].a *= channelWeights[1];
#endif
  maxAlpha = max(maxAlpha, c[1].a);
  c[2] = classify(pixLUT2, data.z, data1.z, preint);
#ifdef LIGHTING
  c[2] = light(pix3dtex, c[2], tc);
#endif
#ifdef CHANNEL_WEIGHTS
  c[2].a *= channelWeights[2];
#endif
  maxAlpha = max(maxAlpha, c[2].a);
#if NUM_CHANNELS == 4
  c[3] = classify(pixLUT2, data.w, data1.w, preint);
#ifdef LIGHTING
  c[3] = light(pix3dtex, c[3], tc);
#endif
#ifdef CHANNEL_WEIGHTS
  c[3].a *= channelWeights[3];
#endif
  maxAlpha = max(maxAlpha, c[3].a);
#endif
#endif

  c[0].rgb *= c[0].a;
  for (int i=1; i<NUM_CHANNELS; ++i)
  {
    c[0].rgb += c[i].rgb * c[i].a;
  }
  c[0].rgb /= maxAlpha;

  gl_FragColor.a = maxAlpha;
  gl_FragColor.rgb = c[0].rgb;
#else
  gl_FragColor = c[0];
#endif

}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
