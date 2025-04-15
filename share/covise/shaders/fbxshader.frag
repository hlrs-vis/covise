#version 450
#pragma import_defines ( VSG_NORMAL, VSG_COLOR, VSG_TEXCOORD0, VSG_LIGHTING, VSG_MATERIAL, VSG_DIFFUSE_MAP, VSG_OPACITY_MAP, VSG_AMBIENT_MAP, VSG_NORMAL_MAP, VSG_SPECULAR_MAP )
#extension GL_ARB_separate_shader_objects : enable
#ifdef VSG_DIFFUSE_MAP
layout(binding = 0) uniform sampler2D diffuseMap;
#endif
#ifdef VSG_OPACITY_MAP
layout(binding = 1) uniform sampler2D opacityMap;
#endif
#ifdef VSG_AMBIENT_MAP
layout(binding = 4) uniform sampler2D ambientMap;
#endif
#ifdef VSG_NORMAL_MAP
layout(binding = 5) uniform sampler2D normalMap;
#endif
#ifdef VSG_SPECULAR_MAP
layout(binding = 6) uniform sampler2D specularMap;
#endif

#ifdef VSG_MATERIAL
layout(binding = 10) uniform MaterialData
{
    vec4 ambientColor;
    vec4 diffuseColor;
    vec4 specularColor;
    float shininess;
} material;
#endif

#ifdef VSG_NORMAL
layout(location = 1) in vec3 normalDir;
#endif
#ifdef VSG_COLOR
layout(location = 3) in vec4 vertColor;
#endif
#ifdef VSG_TEXCOORD0
layout(location = 4) in vec2 texCoord0;
#endif
#ifdef VSG_LIGHTING
layout(location = 5) in vec3 viewDir;
layout(location = 6) in vec3 lightDir;
#endif
layout(location = 0) out vec4 outColor;

void main()
{
#ifdef VSG_DIFFUSE_MAP
    vec4 base = texture(diffuseMap, texCoord0.st);
#else
    vec4 base = vec4(1.0,1.0,1.0,1.0);
#endif
#ifdef VSG_COLOR
    base = base * vertColor;
#endif
#ifdef VSG_MATERIAL
    vec3 ambientColor = material.ambientColor.rgb;
    vec3 diffuseColor = material.diffuseColor.rgb;
    vec3 specularColor = material.specularColor.rgb;
    float shininess = material.shininess;
#else
    vec3 ambientColor = vec3(0.1,0.1,0.1);
    vec3 diffuseColor = vec3(1.0,1.0,1.0);
    vec3 specularColor = vec3(0.3,0.3,0.3);
    float shininess = 16.0;
#endif
#ifdef VSG_AMBIENT_MAP
    ambientColor *= texture(ambientMap, texCoord0.st).r;
#endif
#ifdef VSG_SPECULAR_MAP
    specularColor = texture(specularMap, texCoord0.st).rrr;
#endif
#ifdef VSG_LIGHTING
#ifdef VSG_NORMAL_MAP
    vec3 nDir = texture(normalMap, texCoord0.st).xyz*2.0 - 1.0;
    nDir.g = -nDir.g;
#else
    vec3 nDir = normalDir;
#endif
    vec3 nd = normalize(nDir);
    vec3 ld = normalize(lightDir);
    vec3 vd = normalize(viewDir);
    vec4 color = vec4(0.01, 0.01, 0.01, 1.0);
    color.rgb += ambientColor;
    float diff = max(dot(ld, nd), 0.0);
    color.rgb += diffuseColor * diff;
    color *= base;
    if (diff > 0.0)
    {
        vec3 halfDir = normalize(ld + vd);
        color.rgb += base.a * specularColor *
            pow(max(dot(halfDir, nd), 0.0), shininess);
    }
#else
    vec4 color = base;
    color.rgb *= diffuseColor;
#endif
    outColor = color;
#ifdef VSG_OPACITY_MAP
    outColor.a *= texture(opacityMap, texCoord0.st).r;
#endif

    // crude version of AlphaFunc
    if (outColor.a==0.0) discard;
}
