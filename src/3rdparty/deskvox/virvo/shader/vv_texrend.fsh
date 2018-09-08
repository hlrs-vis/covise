// fragment shader library

// Author: Martin Aumueller <aumueller@hlrs.de>

vec4 classify(sampler2D lut, float s0, float s1, bool preint)
{
#ifdef PREINTEGRATION
  return texture2D(lut, vec2(s0, s1));
#else
  return texture2D(lut, vec2(s0, 0.0));
#endif
}

#define DELTA (0.01)

vec3 gradient(sampler3D tex, vec3 tc)
{
    vec3 sample1;
    vec3 sample2;

    sample1.x = texture3D(tex, tc + vec3(DELTA, 0.0, 0.0)).x;
    sample2.x = texture3D(tex, tc - vec3(DELTA, 0.0, 0.0)).x;
    // signs for y and z are swapped because of texture orientation
    sample1.y = texture3D(tex, tc - vec3(0.0, DELTA, 0.0)).x;
    sample2.y = texture3D(tex, tc + vec3(0.0, DELTA, 0.0)).x;
    sample1.z = texture3D(tex, tc - vec3(0.0, 0.0, DELTA)).x;
    sample2.z = texture3D(tex, tc + vec3(0.0, 0.0, DELTA)).x;

    return sample2.xyz - sample1.xyz;
}


uniform vec3 V;
uniform vec3 lpos;
uniform float constAtt;
uniform float linearAtt;
uniform float quadAtt;
uniform float threshold;

vec4 light(sampler3D tex, vec4 classified, vec3 tc)
{
#ifdef LIGHTING
  const vec3 Ka = vec3(0.3, 0.3, 0.3);
  const vec3 Kd = vec3(0.8, 0.8, 0.8);
  const vec3 Ks = vec3(0.8, 0.8, 0.8);
  const float shininess = 1000.0;

  if (classified.w > threshold)
  {
    vec3 grad = gradient(tex, tc);
    vec3 L = lpos - tc;
    float dist = length(L);
    L /= dist;
    vec3 N = normalize(grad);
    vec3 H = normalize(L + V);

    float att = 1.0 / (constAtt + linearAtt * dist + quadAtt * dist * dist);
    float ldot = dot(L, N.xyz);
    float specular = pow(dot(H, N.xyz), shininess);

    // Ambient term.
    vec3 col = Ka * classified.xyz;

    // Diffuse term.
    col += Kd * ldot * classified.xyz * att;

    // Specular term.
    float spec = pow(dot(H, N), shininess);
    col += Ks * spec * classified.xyz * att;

    return vec4(col, classified.w);
  }
  else
#endif
    return classified;
}
