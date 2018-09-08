#version 120
#extension GL_EXT_geometry_shader4 : enable

uniform vec3 planeNormal;

uniform float delta;
uniform vec3 vertices[8];
uniform vec4 brickMin;
uniform vec3 brickDimInv;
uniform vec3 texMin;
uniform vec3 texRange;

varying in float planeDist[3];

bool testAndSend(const int ori, const int dst)
{
  vec3 vecV1 = vertices[ori];
  vec3 vecV2 = vertices[dst];

  vec3 vecStart = vecV1;
  vec3 vecDir = vecV2-vecV1;

  float denominator = dot(vecDir, planeNormal);
  if (denominator == 0.0)
  {
    return false;
  }
  float lambda = (planeDist[0]-dot(vecStart, planeNormal)) / denominator;

  if ((lambda >= 0.0) && (lambda <= 1.0))
  {
    vec3 pos = vecStart + lambda * vecDir;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1.0);
    gl_TexCoord[0].xyz = (pos - brickMin.xyz) * brickDimInv.xyz;
    gl_TexCoord[0].xyz = gl_TexCoord[0].xyz * texRange + texMin;
    EmitVertex();
    return true;
  }
  else
  {
    return false;
  }
}

void send(const int idx)
{
  gl_Position = gl_PositionIn[idx];
  gl_TexCoord[0] = gl_TexCoordIn[idx][0];
  EmitVertex();
}

void main()
{
  testAndSend(1, 4); // p1
  send(0); // p0
  send(1); // p2
  if (testAndSend(3, 2)) // p5
  {
    testAndSend(5, 6); // p3
    send(2); // p4
  }
  else
  {
    send(2); // p4
    testAndSend(5, 6); // p3
  }
  EndPrimitive();
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
