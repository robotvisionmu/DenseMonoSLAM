#version 440 core

layout(points) in;
layout(points, max_vertices = 1) out;

in vec4 vPosition0[];
in vec4 vColorTime0[];
in vec4 vNormRad0[];
flat in int id[];

layout (location = 0) out vec4 vPointData;
layout (location = 1) out vec4 vNormalData;

uniform int sampleRate;
void main() 
{
    if(id[0] % sampleRate == 0)
    {
        vPointData = vec4(vPosition0[0].xyz, 0);
        vNormalData = vec4(vNormRad0[0].xyz, 0);
        EmitVertex();
        EndPrimitive(); 
    }
}
