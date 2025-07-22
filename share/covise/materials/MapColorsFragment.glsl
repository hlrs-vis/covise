uniform sampler1D texUnit1;
uniform float rangeMin, rangeMax;
uniform bool blendWithMaterial;
uniform bool Light0Enabled, Light1Enabled;
uniform bool dataValid;
#ifdef POINTS
uniform sampler2D diffuseMap;
#endif
varying float data;
varying vec3 V;
varying vec3 N;

vec4 Ambient;
vec4 Diffuse;
vec4 Specular;

void directionalLight(in int i, in vec3 normal) {
    float nDotVP;         // normal . light direction
    float nDotHV;         // normal . light half vector
    float pf;             // power factor

    nDotVP = abs(dot(normal, normalize(vec3(gl_LightSource[i].position))));
    nDotHV = abs(dot(normal, vec3(gl_LightSource[i].halfVector)));

    if (nDotVP == 0.0) {
        pf = 0.0;
    }
    else {
        pf = pow(nDotHV, gl_FrontMaterial.shininess);
    }
    Ambient  += gl_LightSource[i].ambient;
    Diffuse  += gl_LightSource[i].diffuse * nDotVP;
    Specular += gl_LightSource[i].specular * pf;
}

void pointLight(in int i, in vec3 N, in vec3 eye) {
    float pf;           // power factor
    float d;            // distance from surface to light source

    vec3 L = normalize(gl_LightSource[i].position.xyz - V);
    float NdotL = abs(dot(N, L));
    if (NdotL == 0.0) {
        pf = 0.0;
    }
    else {
        vec3 E = normalize(-V);
        vec3 R = normalize(-reflect(L, N));
        pf = pow(abs(dot(R, E)), gl_FrontMaterial.shininess);
    }

    // Compute distance between surface and light position
    d = length(L);

    // Compute attenuation
    float attenuation = 1.0 / (gl_LightSource[i].constantAttenuation +
        gl_LightSource[i].linearAttenuation * d +
        gl_LightSource[i].quadraticAttenuation * d * d);

    Ambient  += gl_LightSource[i].ambient * attenuation;
    Diffuse  += gl_LightSource[i].diffuse * NdotL * attenuation;
    Specular += gl_LightSource[i].specular * pf * attenuation;
}

vec4 flight(in vec3 normal, vec4 color) {
    vec3 eye = vec3 (0.0, 0.0, 1.0);

    // Clear the light intensity accumulators
    Ambient  = vec4 (0.0);
    Diffuse  = vec4 (0.0);
    Specular = vec4 (0.0);

    if (Light0Enabled)
    pointLight(0, normal, eye);
    if (Light1Enabled)
    pointLight(1, normal, eye);

    //directionalLight(2, normal);

    return color * (Diffuse + Ambient) + Specular * color.a;
}

vec4 flightm(in vec3 normal) {
    vec3 eye = vec3(0.0, 0.0, 1.0);

    // Clear the light intensity accumulators
    Ambient = vec4(0.0);
    Diffuse = vec4(0.0);
    Specular = vec4(0.0);

    if (Light0Enabled)
    pointLight(0, normal, eye);
    if (Light1Enabled)
    pointLight(1, normal, eye);

    //directionalLight(2, normal);

    return gl_FrontMaterial.diffuse * Diffuse + gl_FrontMaterial.ambient * Ambient +
    gl_FrontMaterial.specular * Specular;
}

void main (void) {
    if (dataValid) {
        float t = (data - rangeMin) / (rangeMax - rangeMin);
        // t = clamp(t, 0.0, 1.0); // Ensure t is in [0,1]
        vec4 color = texture1D(texUnit1, t);
        if (blendWithMaterial) {
            color = mix(gl_FrontMaterial.diffuse, color, color.a);
            color.a = 1.0;
        }
        gl_FragColor = flight(normalize(N), color);

        if (!blendWithMaterial) {
            gl_FragColor.a = color.a;
        }
    }
    else {
        gl_FragColor = flightm(normalize(N));
    }
#ifdef POINTS
    gl_FragColor = gl_FragColor * texture2D(diffuseMap , gl_TexCoord[0].xy);
#endif
}

// void main(void) {
//     // float t = (data - rangeMin) / (rangeMax - rangeMin);
//     // t = clamp(t, 0.0, 1.0); // Ensure t is in [0,1]
//     // gl_FragColor = vec4(t, t, t, 1.0); // Grayscale output
//     if (data > 1.0)
//     gl_FragColor = vec4(1,0,0,1); // Red if data > 10
//     else
//     gl_FragColor = vec4(0,0,1,1); // Blue otherwise
// }