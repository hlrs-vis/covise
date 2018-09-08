#version 120

varying mat4 P;
varying mat4 Rot;
varying float psize;

void main(void)
{

  vec2 v2;
  v2.xy = gl_PointCoord * 2.0 - vec2(1.0);
  v2.x *= psize;
  v2.y *= psize;

  vec4 v4 = vec4(v2.x, v2.y, 0., 1.);
  v4 *= Rot;
  v4 *= P;

  float mag = dot(v4.xy, v4.xy);
  if (mag > 1.)
    discard;

  gl_FragColor = gl_Color;
}
