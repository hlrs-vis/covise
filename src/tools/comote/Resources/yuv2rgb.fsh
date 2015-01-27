// YUV to RGB pixel shader. Loads a pixel from each plane and pass through the
// matrix.

uniform sampler2D y_tex;
uniform sampler2D u_tex;
uniform sampler2D v_tex;

void main() {
    vec2 tc = gl_TexCoord[0].xy;
    vec3 yuv = vec3(texture2D(y_tex, tc).r,
                    texture2D(u_tex, tc).r,
                    texture2D(v_tex, tc).r);
    
    gl_FragColor = vec4(yuv.x + 1.403 * yuv.z - 0.702,
                        yuv.x - 0.344 * yuv.y - 0.714 * yuv.z + 0.529,
                        yuv.x + 1.772 * yuv.y - 0.886,
                        1.);
}
