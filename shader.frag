#version 450
#extension GL_ARB_separate_shader_objects : enable

//Image sampler for textures
layout(binding = 1) uniform sampler2D texSampler;

//The input coming from the vertex shader for
//colors and textures
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

//Outputing the final colors 
layout(location = 0) out vec4 outColor;

void main() 
{
    //Sending out the texture with sampler, and
    //coords as params
    outColor = texture(texSampler, fragTexCoord);
}