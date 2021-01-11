//This file is written in glsl and functions as
//the vertex shader
#version 450
#extension GL_ARB_separate_shader_objects : enable

//outputting colors
layout(location = 0) out vec3 fragColor;

//2 points for a triangle
vec2 positions[3] = vec2[]
(
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

//3 volors for this triangle
vec3 colors[3] = vec3[]
(
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() 
{
    //Outputting position
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);

    //Assigning colors to an output
    //vector var
    fragColor = colors[gl_VertexIndex];
}