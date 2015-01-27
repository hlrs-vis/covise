
const float PI = 3.14159;
const float ONE_OVER_PI = 0.31831;

varying vec3 N, L, H, R, T, B;               // Variables for calculating the brdf
varying vec3 BaseColor;                      // Color of the glass
varying float Transparency;

uniform float Brightness;
uniform vec2 Reflectance;
uniform vec2 Slope;
uniform vec3 ScaleFactors;
uniform float CubemapRatio;                  // Ratio of the color and the texture lookup
uniform samplerCube EnvMap;                  // Environmentmap

uniform sampler2D EXISTING_TEXTURE;

void main()
{

   float e1, e2, E, cosThetaI, cosThetaR, brdf, intensity;

   e1 = dot(H, T) / Slope.x;
   e2 = dot(H, B) / Slope.y;
   E = -1.0 * ((e1 * e1 + e2 * e2) / (1.0 + dot(H, N)));

   cosThetaI = abs(dot(N, L));
   cosThetaR = abs(dot(N, R));

   brdf = Reflectance.x * ONE_OVER_PI +
          Reflectance.y * (1.0 / (sqrt(cosThetaI * cosThetaR) * (4.0 * PI * Slope.x * Slope.y))) * exp(E);

   intensity = ScaleFactors[0] * Reflectance.x * ONE_OVER_PI +        // diffuse
               ScaleFactors[1] * Reflectance.y * cosThetaI * brdf +   // brdf
               ScaleFactors[2] * dot(H, N) * Reflectance.y;           // specular
   intensity = intensity * Brightness;

   // Use existing texture
   vec4 tex = texture2D(EXISTING_TEXTURE, gl_TexCoord[0].st);
   if (tex.r < 0.001f && tex.g < 0.001f && tex.b < 0.001f)
   {
      tex.a = 0.0f;
   }
   vec3 color = mix(BaseColor*intensity, tex.rgb*intensity, tex.a);

   // Mix cubemap
   vec3 envColor = vec3(textureCube(EnvMap, R));
   color = mix(color, envColor, CubemapRatio);

   gl_FragColor = vec4(color, Transparency);
}
