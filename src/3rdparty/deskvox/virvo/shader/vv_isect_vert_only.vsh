uniform vec3 planeNormal;

uniform float delta;
uniform vec3 vertices[8];
uniform vec4 brickMin;
uniform vec3 brickDimInv;
uniform vec3 texMin;
uniform vec3 texRange;
uniform int v1[24];

void main()
{
  float planeDist = brickMin.w + gl_Vertex.y * delta;
  vec3 pos;

  vec3 vecV1 = vertices[v1[int(gl_Vertex.x)]];
  for (int i=0; i<3; ++i)
  {
    vec3 vecV2 = vertices[v1[int(gl_Vertex.x) + i + 1]];

    vec3 vecStart = vecV1;
    vec3 vecDir = vecV2-vecV1;

    float denominator = dot(vecDir, planeNormal);
    float lambda = (denominator != 0.0) ?
                      (planeDist-dot(vecStart, planeNormal)) / denominator
                   :
                     -1.0;

    if ((lambda >= 0.0) && (lambda <= 1.0))
    {
      pos = vecStart + lambda * vecDir;
      break;
    }
    vecV1 = vecV2;
  }

  gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1.0);
  gl_TexCoord[0].xyz = (pos - brickMin.xyz) * brickDimInv.xyz;
  gl_TexCoord[0].xyz = gl_TexCoord[0].xyz * texRange + texMin;
  gl_FrontColor = vec4(1, 1, 1, 1);
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
