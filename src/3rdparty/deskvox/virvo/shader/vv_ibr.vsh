#version 120

uniform mat4 reprojectionMatrix;
uniform sampler2D rgbaTex;
uniform sampler2D depthTex;
uniform float imageWidth;
uniform float imageHeight;
uniform float vpWidth;
uniform float vpHeight;
uniform float splitX, splitY;
uniform float depthMin;
uniform float depthRange;
uniform bool closer;

uniform mat4 V_i;
uniform float si;
uniform float sj;
uniform float sk;

varying mat4 P;
varying mat4 Rot;
varying float theta;
varying float psize;

void main(void)
{
  float x = gl_Vertex.x;
  float y = gl_Vertex.y;
  if(closer)
  {
    // start at projection center
    if(gl_Vertex.x < splitX)
      x = floor(splitX)-gl_Vertex.x;
    if(gl_Vertex.y < splitY)
      y = floor(splitY)-gl_Vertex.y;
  }
  else
  {
    // proceed outside-in
    if(gl_Vertex.x > splitX)
      x = imageWidth+floor(splitX)-gl_Vertex.x;
    if(gl_Vertex.y > splitY)
      y = imageHeight+floor(splitY)-gl_Vertex.y;
  }
  vec2 tc = vec2(x/imageWidth, y/imageHeight);
  gl_FrontColor = texture2D(rgbaTex, tc);
  vec4 c = gl_FrontColor;
  float d = texture2D(depthTex, tc).r;
  // p will be in normalized device coordinates
  vec4 p = vec4(tc.x, tc.y, depthMin+d*depthRange, 1.);
  p.xyz *= 2.;
  p.xyz -= vec3(1., 1., 1.);
  gl_Position = reprojectionMatrix * p;
  if(d <= 0.)
  {
    gl_Position.z = gl_Position.w * 1.1; // clip vertex
  }
  else
  {
#if 0
    p += vec4(2./imageWidth, 2./imageHeight, 0., 0.);
    vec4 s1 = gl_Position;
    vec4 s2 = reprojectionMatrix * p;
    s1 /= s1.w;
    s1 *= vpWidth/imageWidth;
    s2 /= s2.w;
    s2 *= vpHeight/imageHeight;
    vec2 d = vec2((s1.x-s2.x)*vpWidth, (s1.y-s2.y)*vpHeight);
    gl_PointSize = length(d) * 0.7; // 0.7 = 1/sqrt(2)
#else
    // See 'Footprint Evaluation for Volume Rendering, Westover, L. 1990'
    // and 'Splatting of Curvilinear Volumes, Xiaoyang, M. et al. 1995'

    // z/2 is a hack for now.
    mat4 mE = mat4(si, 0., 0., 0.,
                  0., sj, 0., 0.,
                  0., 0., sk*.5, 0.,
                  0., 0., 0., -1.);

    mat4 V_it;
    //transpose(V_it); // doesn't work with fedora 14 somehow
    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        V_it[i][j] = V_i[j][i];
      }
    }

    mat4 R = V_it * mE * V_i;
    float A = R[0][0];
    float B = R[1][1];
    float C = R[2][2];
    float D = R[0][1] * 2.;
    float E = R[0][2] * 2.;
    float F = R[1][2] * 2.;

    // screen space ellipse in quadratic form: Xx^2 + Yy^2 + Zxy = K
    float X = A - ((E*E)/(4.*C));
    float Y = B - ((F*F)/(4.*C));
    float Z = D - ((E*F)/(2.*C));

    theta = 0.;
    float S_x = 0.;
    float S_y = 0.;

    if (Z == 0.)
    {
      theta = 0.;
      S_x = sqrt(X);
      S_y = sqrt(Y);
    }
    else
    {
      theta = .5 * atan(Z / (X - Y));
      if (theta == 0.) theta = .01;
      S_x = sqrt((X + Y + (Z / sin(2. * theta))) / 2.);
      S_y = sqrt((X + Y - (Z / sin(2. * theta))) / 2.);
    }

    mat4 T = mat4(1.);
    T[0][0] = 1. / S_x;
    T[1][1] = 1. / S_y;

    mat4 Tt;

    //transpose(Tt);
    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        Tt = T;
      }
    }
    
    P = T * Tt;

    float S_x2 = S_x*S_x;
    float S_y2 = S_y*S_y;

    Rot = mat4(1.);
    if (Z != 0.)
    {
      Rot[0][0] = cos(theta);
      Rot[0][1] = -sin(theta);
      Rot[1][0] = sin(theta);
      Rot[1][1] = cos(theta);

      vec4 left =   vec4(-S_x2, -S_y2, 0., 1.);
      vec4 right =  vec4( S_x2, -S_y2, 0., 1.);
      vec4 top =    vec4( S_x2,  S_y2, 0., 1.);
      vec4 bottom = vec4(-S_x2,  S_y2, 0., 1.);

      left *= Rot;
      right *= Rot;
      top *= Rot;
      bottom *= Rot;

      float minX = left.x;
      float maxX = left.x;
      float minY = left.y;
      float maxY = left.y;

      if (right.x < minX) minX = right.x;
      if (right.x > maxX) maxX = right.x;
      if (right.y < minY) minY = right.y;
      if (right.y > maxY) maxY = right.y;
      
      if (top.x < minX) minX = top.x;
      if (top.x > maxX) maxX = top.x;
      if (top.y < minY) minY = top.y;
      if (top.y > maxY) maxY = top.y;
      
      if (bottom.x < minX) minX = bottom.x;
      if (bottom.x > maxX) maxX = bottom.x;
      if (bottom.y < minY) minY = bottom.y;
      if (bottom.y > maxY) maxY = bottom.y;

      float psx = maxX - minX;
      float psy = maxY - minY;

      gl_PointSize = psx >= psy ? psx : psy;
    }
    else
    {
      gl_PointSize = (S_x2 >= S_y2) ? 2. * S_x2 :  2. * S_y2;
    }
    psize = gl_PointSize / 2.;
#endif
  }
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
