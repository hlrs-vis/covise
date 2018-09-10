uniform vec3 planeNormal;

uniform float delta;
uniform vec3 vertices[8];
uniform vec4 brickMin;
uniform vec3 brickDimInv;
uniform vec3 texMin;
uniform vec3 texRange;
uniform int v1[9];
uniform int v2[9];

varying float planeDist;

void main()
{
  planeDist = brickMin.w + gl_Vertex.y * delta;
  vec3 pos;
  
  for (int i=0; i<3; ++i)
  {
    int vIdx1 = v1[int(gl_Vertex.x) + i];
    int vIdx2 = v2[int(gl_Vertex.x) + i];
    
    vec3 vecV1 = vertices[vIdx1];
    vec3 vecV2 = vertices[vIdx2];

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
  }
  
  gl_Position = gl_ModelViewProjectionMatrix * vec4(pos, 1.0);
  gl_TexCoord[0].xyz = (pos - brickMin.xyz) * brickDimInv.xyz;
  gl_TexCoord[0].xyz = gl_TexCoord[0].xyz * texRange + texMin;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
