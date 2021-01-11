//This file is written in glsl and functions as the
//fragment shader.
#version 450
#extension GL_ARB_separate_shader_objects : enable

//Loading in color vector
layout(location = 0) in vec3 fragColor;
//Outputting colored fragments
layout(location = 0) out vec4 outColor;

void main() 
{
    outColor = vec4(fragColor, 1.0);
}