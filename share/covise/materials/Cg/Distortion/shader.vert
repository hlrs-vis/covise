varying vec2 texCoord;

void main()
{
	texCoord = gl_MultiTexCoord0.st;	//gl_MultiTexCoord0 liefert Texturkoordinaten in vec4 (s,t,p,q) zurück -> siehe GLSL-Ref
	gl_Position = ftransform();			//transformiert Eingangs-Vertex wie es die fixed-function-pipeline tun würde -> (projection_matrix * modelview_matrix * vertex)
}
