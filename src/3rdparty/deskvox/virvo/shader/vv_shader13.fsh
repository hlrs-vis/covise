// Author of cg-version: 
//  Stefan Zellmann <zellmans@uni-koeln.de>
//
// Converted by:
//  Stavros Delisavas <stavros.delisavas@uni-koeln.de>

#define DELTA (0.01)

uniform sampler3D pix3dtex;
uniform sampler2D pixLUT;
uniform vec3 V;
uniform vec3 lpos;
uniform float constAtt;
uniform float linearAtt;
uniform float quadAtt;
uniform float threshold;

void main()
{
  // TODO: make these parameters uniform and configurable.
  const vec3 Ka = vec3(0.0, 0.0, 0.0);
  const vec3 Kd = vec3(0.8, 0.8, 0.8);
  const vec3 Ks = vec3(0.8, 0.8, 0.8);
  const float shininess = 1000.0;

  vec4 center;
  float centerwx = texture3D(pix3dtex, gl_TexCoord[0].xyz).x;
  center.w = centerwx;
  center.x = centerwx;
  vec4 classification = texture2D(pixLUT, center.wx);

  vec4 OUT;
  if (classification.w > threshold)
  {
    vec3 sample1;
    vec3 sample2;
    sample1.x = texture3D(pix3dtex, gl_TexCoord[0].xyz - vec3(DELTA, 0.0, 0.0)).x;
    sample2.x = texture3D(pix3dtex, gl_TexCoord[0].xyz + vec3(DELTA, 0.0, 0.0)).x;
    sample1.y = texture3D(pix3dtex, gl_TexCoord[0].xyz - vec3(0.0, DELTA, 0.0)).x;
    sample2.y = texture3D(pix3dtex, gl_TexCoord[0].xyz + vec3(0.0, DELTA, 0.0)).x;
    sample1.z = texture3D(pix3dtex, gl_TexCoord[0].xyz - vec3(0.0, 0.0, DELTA)).x;
    sample2.z = texture3D(pix3dtex, gl_TexCoord[0].xyz + vec3(0.0, 0.0, DELTA)).x;

    vec3 L = lpos - gl_TexCoord[0].xyz;
    vec3 N = normalize(sample2.xyz - sample1.xyz);
    N.x = -N.x; // mind volume coordinate system
    float dist = length(L);
    L = normalize(L);
    vec3 H = normalize(L + V);

    float att = 1.0 / (constAtt + linearAtt * dist + quadAtt * dist * dist);
    float ldot = dot(L, N.xyz);
    float specular = pow(dot(H, N.xyz), shininess);

    // Ambient term.
    OUT.xyz = Ka * classification.xyz;

    // Diffuse term.
    OUT.xyz += Kd * ldot * classification.xyz * att;

    // Specular term.
    float spec = pow(dot(H, N), shininess);

    OUT.xyz += Ks * spec * classification.xyz * att;
    OUT.w = classification.w;
    gl_FragColor = OUT;
  }
  else
  {
    gl_FragColor = classification;
  }
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
