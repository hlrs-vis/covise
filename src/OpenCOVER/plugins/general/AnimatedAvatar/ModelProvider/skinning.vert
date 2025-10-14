/*  -*-c++-*- 
 *  Copyright (C) 2008 Cedric Pinson <cedric.pinson@plopbyte.net>
 *
 * This library is open source and may be redistributed and/or modified under  
 * the terms of the OpenSceneGraph Public License (OSGPL) version 0.0 or 
 * (at your option) any later version.  The full license is in LICENSE file
 * included with this distribution, and on the openscenegraph.org website.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * OpenSceneGraph Public License for more details.
*/

in vec4 boneWeight0;
in vec4 boneWeight1;
in vec4 boneWeight2;
in vec4 boneWeight3;

uniform uint nbBonesPerVertex;
uniform mat4 matrixPalette[MAX_MATRIX];

vec4 position;
vec3 normal;


// accumulate position and normal in global scope
void computeAcummulatedNormalAndPosition(vec4 boneWeight)
{
    int matrixIndex;
    float matrixWeight;
    for (int i = 0; i < 2; i++)
    {
        matrixIndex =  int(boneWeight[0]);
        matrixWeight = boneWeight[1];
        mat4 matrix = matrixPalette[matrixIndex];
        // correct for normal if no scale in bone
        mat3 matrixNormal = mat3(matrix);
        position += matrixWeight * (matrix * gl_Vertex );
        normal += matrixWeight * (matrixNormal * gl_Normal );

        boneWeight = boneWeight.zwxy;
    }
}

void main( void )
{
    position =  vec4(0.0,0.0,0.0,0.0);
		normal = vec3(0.0,0.0,0.0);
    gl_TexCoord[0] = gl_MultiTexCoord0;
    // there is 2 bone data per attributes
    if (nbBonesPerVertex > 0u)
        computeAcummulatedNormalAndPosition(boneWeight0);
    if (nbBonesPerVertex > 2u)
        computeAcummulatedNormalAndPosition(boneWeight1);
    if (nbBonesPerVertex > 4u)
        computeAcummulatedNormalAndPosition(boneWeight2);
    if (nbBonesPerVertex > 6u)
        computeAcummulatedNormalAndPosition(boneWeight3);

    normal = gl_NormalMatrix * normal;

    vec3 lightDir = normalize(vec3(gl_LightSource[0].position));
    float NdotL = max(dot(normal, lightDir), 0.0);
		vec4 diffuse = NdotL * gl_FrontMaterial.diffuse * gl_LightSource[0].diffuse;

		vec4 ambient = gl_FrontMaterial.ambient * gl_LightSource[0].ambient;
		vec4 globalAmbient = gl_LightModel.ambient * gl_FrontMaterial.ambient;

		float NdotHV = max(dot(normal, gl_LightSource[0].halfVector.xyz),0.0);
		vec4 specular = gl_FrontMaterial.specular * gl_LightSource[0].specular * pow(NdotHV,gl_FrontMaterial.shininess);

		gl_FrontColor = specular + diffuse + globalAmbient + ambient;
    gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * position;
}
