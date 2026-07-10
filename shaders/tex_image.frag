#version 450
layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragCol;

layout (set=0, binding=0) uniform sampler2D displayTexture;

void main(){
	outFragCol = texture(displayTexture, inUV);
}
