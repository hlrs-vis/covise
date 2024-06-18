#if __VERSION__ <= 120
#extension GL_EXT_geometry_shader4 : enable
#endif
#extension GL_ARB_texture_rectangle : enable

#if __VERSION__ > 120
layout (points) in;
layout (triangle_strip, max_vertices = 24) out;
#endif

uniform sampler2DRect heightTex;
uniform bool dataValid;
uniform sampler2DRect dataTex;
uniform vec2 size;
uniform vec2 dist;
uniform vec4 origin;
uniform vec2 patchSize;

#if __VERSION__ > 120
out vec3 N;
out vec3 V;
out float data;
#else
varying out vec3 N;
varying out vec3 V;
varying out float data;
#endif

#define CACHE 1

#if CACHE
#define MaxPatchSizeY 3
vec4 posCache[MaxPatchSizeY+1];
vec4 norm_dataCache[MaxPatchSizeY+1];
#endif

float getHeight(vec2 xy)
{
    return texture2DRect(heightTex, xy + vec2(.5, .5)).r;
}

float getData(vec2 xy)
{
    return texture2DRect(dataTex, xy + vec2(.5, .5)).r;
}

vec4 pos(vec2 xy, float h)
{
    vec4 p = origin + vec4(dist * xy, h, 0.);
    return p;
}

#if CACHE
void useVertex(int idx)
{
    gl_Position = posCache[idx];
    gl_ClipVertex = gl_Position;
    V = gl_Position.xyz / gl_Position.w;
    N = norm_dataCache[idx].xyz;
    data = norm_dataCache[idx].w;
    EmitVertex();
}
#endif

void createVertex(vec2 xy, int idx)
{
    float h = getHeight(xy);
    gl_Position = gl_ModelViewProjectionMatrix * pos(xy, h);
#if CACHE
    if (idx >= 0)
        posCache[idx] = gl_Position;
#endif
    gl_ClipVertex = gl_Position;
    V = gl_Position.xyz / gl_Position.w;

    float dx = 0., dy = 0.;
    float hn = h, hs = h, he = h, hw = h;
    if (xy.x >= 1)
    {
        he = getHeight(xy - vec2(1, 0));
        dx += dist.x;
    }
    if (xy.x < size.x)
    {
        hw = getHeight(xy + vec2(1, 0));
        dx += dist.x;
    }
    if (xy.y >= 1)
    {
        hs = getHeight(xy + vec2(0, 1));
        dy += dist.y;
    }
    if (xy.y < size.y)
    {
        hn = getHeight(xy + vec2(0, 1));
        dy += dist.y;
    }
    N = normalize(gl_NormalMatrix * vec3((hw - he) / dx, (hn - hs) / dy, 1.));
#if CACHE
    if (idx >= 0)
        norm_dataCache[idx].xyz = N;
#endif

    if (dataValid)
        data = getData(xy);
    else
        data = 0.;
#if CACHE
    if (idx >= 0)
        norm_dataCache[idx].w = data;
#endif
    EmitVertex();
}

void main()
{
#if __VERSION__ > 120
    vec2 xy = gl_in[0].gl_Position.xy;
#else
    vec2 xy = gl_PositionIn[0].xy;
#endif

    createVertex(xy + vec2(0, 0), -1);
    createVertex(xy + vec2(1, 0), 0);
    for (int y = 1; y < patchSize.y + 1; ++y)
    {
        if (xy.y + y == size.y)
            break;
        createVertex(xy + vec2(0, y), -1);
        createVertex(xy + vec2(1, y), y);
    }
    EndPrimitive();

    for (int x = 1; x < patchSize.x; ++x)
    {
        if (xy.x + x == size.x - 1)
            break;
#if CACHE
        useVertex(0);
#else
        createVertex(xy + vec2(x, 0), -1);
#endif
        createVertex(xy + vec2(x + 1, 0), 0);
        for (int y = 1; y < patchSize.y + 1; ++y)
        {
            if (xy.y + y == size.y)
                break;
#if CACHE
            useVertex(y);
#else
            createVertex(xy + vec2(x, y), -1);
#endif
            createVertex(xy + vec2(x + 1, y), y);
        }
        EndPrimitive();
    }
}
