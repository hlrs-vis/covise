#version 450
#pragma import_defines ( VSG_NORMAL, VSG_TANGENT, VSG_COLOR, VSG_TEXCOORD0, VSG_LIGHTING, VSG_NORMAL_MAP, VSG_BILLBOARD, VSG_TRANSLATE )
#extension GL_ARB_separate_shader_objects : enable
layout(push_constant) uniform PushConstants {
    mat4 projection;
    mat4 modelView;
    //mat3 normal;
} pc;
layout(location = 0) in vec3 osg_Vertex;
#ifdef VSG_NORMAL
layout(location = 1) in vec3 osg_Normal;
layout(location = 1) out vec3 normalDir;
#endif
#ifdef VSG_TANGENT
layout(location = 2) in vec4 osg_Tangent;
#endif
#ifdef VSG_COLOR
layout(location = 3) in vec4 osg_Color;
layout(location = 3) out vec4 vertColor;
#endif
#ifdef VSG_TEXCOORD0
layout(location = 4) in vec2 osg_MultiTexCoord0;
layout(location = 4) out vec2 texCoord0;
#endif
#ifdef VSG_LIGHTING
layout(location = 5) out vec3 viewDir;
layout(location = 6) out vec3 lightDir;
#endif
#ifdef VSG_TRANSLATE
layout(location = 7) in vec3 translate;
#endif


out gl_PerVertex{ vec4 gl_Position; };

void main()
{
    mat4 modelView = pc.modelView;

#ifdef VSG_TRANSLATE
    mat4 translate_mat = mat4(1.0, 0.0, 0.0, 0.0,
                              0.0, 1.0, 0.0, 0.0,
                              0.0, 0.0, 1.0, 0.0,
                              translate.x,  translate.y,  translate.z, 1.0);

    modelView = modelView * translate_mat;
#endif

#ifdef VSG_BILLBOARD
    vec3 lookDir = vec3(-modelView[0][2], -modelView[1][2], -modelView[2][2]);

    // rotate around local z axis
    float l = length(lookDir.xy);
    if (l>0.0)
    {
        float inv = 1.0/l;
        float c = lookDir.y * inv;
        float s = lookDir.x * inv;

        mat4 rotation_z = mat4(c,   -s,  0.0, 0.0,
                               s,   c,   0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0);

        modelView = modelView * rotation_z;
    }
#endif

    gl_Position = (pc.projection * modelView) * vec4(osg_Vertex, 1.0);

#ifdef VSG_TEXCOORD0
    texCoord0 = osg_MultiTexCoord0.st;
#endif
#ifdef VSG_NORMAL
    vec3 n = (modelView * vec4(osg_Normal, 0.0)).xyz;
    normalDir = n;
#endif
#ifdef VSG_LIGHTING
    vec4 lpos = /*osg_LightSource.position*/ vec4(0.0, 0.25, 1.0, 1.0);
#ifdef VSG_NORMAL_MAP
    vec3 t = (modelView * vec4(osg_Tangent.xyz, 0.0)).xyz;
    vec3 b = cross(n, t);
    vec3 dir = -vec3(modelView * vec4(osg_Vertex, 1.0));
    viewDir.x = dot(dir, t);
    viewDir.y = dot(dir, b);
    viewDir.z = dot(dir, n);
    if (lpos.w == 0.0)
        dir = lpos.xyz;
    else
        dir += lpos.xyz;
    lightDir.x = dot(dir, t);
    lightDir.y = dot(dir, b);
    lightDir.z = dot(dir, n);
#else
    viewDir = -vec3(modelView * vec4(osg_Vertex, 1.0));
    if (lpos.w == 0.0)
        lightDir = lpos.xyz;
    else
        lightDir = lpos.xyz + viewDir;
#endif
#endif
#ifdef VSG_COLOR
    vertColor = osg_Color;
#endif
}
