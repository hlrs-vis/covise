uniform sampler2D textureDistImg;		//Textur mit Image der Distortion-Map
uniform sampler2D textureDistort;		//Textur mit gerendertem Subgraph (gilt es zu verzerren)
uniform sampler2D textureBlendImg;		//Textur mit edge-Blending Image 
uniform bool blend;						//EdgeBlending ja/nein
uniform bool distort;					//Verzerren ja/nein
varying vec2 texCoord;					//Übergabe der aktuellen Texturkoordinaten vom Vertex-Shader

//ACHTUNG: Farbwerte hier von 0(min) bis 1 (max)

void main()
{
	vec4 distImgColor;	//Farbwerte (r,g,b,a) aus DistortionImage an der Stelle der akt. Texturkoordinate
    vec2 texCoordDist;	//Über distImg ausgelesene, verschobene Texturkoordinaten 
    vec4 distortColor;	//Farbwerte (r,g,b,a) an der Stelle der aktuellen, bzw der ausgelesenen verschobenen Texturkoordinaten
    vec4 blendColor;	//Farbwert aus edge-Blending Image (s/w) an der Stelle der akt. Texturkoordinaten
    vec4 finalColor;	//Farbwert der schlussendlich an akt. Texturkoordinate ausgegeben wird						

	if (blend) 
	{
		// schwarz->finalColor wird schwarz, weiß->finalColor bleib unverändert
		blendColor = texture2D(textureBlendImg, texCoord);
	}
	else
	{
		// wird auf weiß gesetzt -> finalColor unverändert
		blendColor = vec4(1.0f,1.0f,1.0f,1.0f);
	}
    
	if (distort)
	{
		distImgColor = texture2D(textureDistImg, texCoord);
	    texCoordDist[0] = (distImgColor.r + mod(floor(distImgColor.b * 255), 16.0)) / 16.0;		
	    texCoordDist[1] = (distImgColor.g + floor(distImgColor.b * 255 / 16.0)) / 16.0;
		
		//Farbe an der Stelle der verschobenen Texturkoordinate des Subgraph
		distortColor = texture2D(textureDistort, texCoordDist);
	}		
    else
    {
		//Farbe an der Stelle der akutellen unverschobenen Texturkoordinate des Subgraph
		distortColor = texture2D(textureDistort, texCoord);
    }
    
    finalColor = distortColor * blendColor;
    
    //Farbeausgabe an Position der aktuellen Texturkoordinate
    gl_FragColor = finalColor;		
}
