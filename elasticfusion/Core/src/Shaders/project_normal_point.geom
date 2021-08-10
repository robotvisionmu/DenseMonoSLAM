#version 440 core

layout(points) in;
layout(points, max_vertices = 1) out;

in vec4 vPosition[];
in vec4 vNormal[];

out vec4 vPosition0;
out vec4 vNormal0;


void main()
{
    if(vPosition[0].z != -1)
    {
        vPosition0 = vPosition[0];
        vNormal0 = vNormal[0];

        EmitVertex();
        EndPrimitive();
    }
}