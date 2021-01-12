//This file is written in glsl and functions as
//the vertex shader
#version 450
#extension GL_ARB_separate_shader_objects : enable

//Vertex attributes-specified per vertex
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

//Applied to all vertices
void main() 
{
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}
