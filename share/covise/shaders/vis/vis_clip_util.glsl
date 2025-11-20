#ifndef VIS_CLIP_UTIL_GLSL
#define VIS_CLIP_UTIL_GLSL

const int LAMURE_MAX_CLIP_PLANES = 6;

void lamure_apply_clip_planes(vec4 world_pos,
                              int planeCount,
                              vec4 clipPlanes[LAMURE_MAX_CLIP_PLANES])
{
    int limited = min(planeCount, LAMURE_MAX_CLIP_PLANES);
    for (int i = 0; i < limited; ++i)
    {
        gl_ClipDistance[i] = dot(clipPlanes[i], world_pos);
    }
    for (int i = limited; i < LAMURE_MAX_CLIP_PLANES; ++i)
    {
        gl_ClipDistance[i] = 0.0;
    }
}

#endif // VIS_CLIP_UTIL_GLSL
