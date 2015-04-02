uniform sampler2D normalDepthMap;
uniform sampler2D edgeMap;
uniform float first;
uniform float width;
uniform float height;

void main( void )
{
    vec2 dx = vec2(1.0/width, 0.0);
    vec2 dy = vec2(0.0, 1.0/height);
    //vec2 center = gl_TexCoord[1].st;
    vec2 center = vec2(gl_FragCoord.x/width, gl_FragCoord.y/height);
    
    vec4 ul, ur, ll, lr;
    
    ul = texture2D(normalDepthMap, center -dx + dy);
    ur = texture2D(normalDepthMap, center + dx + dy);
    ll = texture2D(normalDepthMap, center - dx + dy);
    lr = texture2D(normalDepthMap, center + dx - dy);
    
    // calculate discontinuities
    vec3 discontinuity = vec3(0.0, 0.0, 0.0);
    
    //(Maybe should decode normals from [0,1] to [-1, 1], but works well the way it is done below)
    float dot0 = dot(ul.xyz, lr.xyz);
    float dot1 = dot(ur.xyz, ll.xyz);
    discontinuity.x = 0.5*(dot0+dot1);
    
    float depth_discont0 = 1.0-abs(ul.w - lr.w);
    float depth_discont1 = 1.0-abs(ur.w - ll.w);
	  
    discontinuity.y = depth_discont0*depth_discont1;
    
    discontinuity.z = abs(discontinuity.x*discontinuity.y);
    
    //combine the generated edge map with the previous edge map
    vec4 prevEdge = texture2D(edgeMap, center);
    discontinuity.z = max(discontinuity.z - (1.0-first)*(1.0 - prevEdge.z), 0.0);
    gl_FragColor = vec4(discontinuity.z, discontinuity.z, discontinuity.z, 1.0);
    
}
