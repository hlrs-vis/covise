
// ---------- vis_surfel_pass1.glslf ----------
#version 420 core

uniform mat4 projection_matrix;

in GsOut {
    noperspective vec2 uv;
    flat vec3 vs_center;
    flat vec3 vs_half_u;
    flat vec3 vs_half_v;
} fs_in;

void main() {
    if (dot(fs_in.uv, fs_in.uv) > 1.0) discard;

    vec3 pos_vs = fs_in.vs_center
                + fs_in.vs_half_u * fs_in.uv.x
                + fs_in.vs_half_v * fs_in.uv.y;

    vec4 clip = projection_matrix * vec4(pos_vs, 1.0);
    float z01 = clip.z / clip.w * 0.5 + 0.5;
    gl_FragDepth = z01;
}