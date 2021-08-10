#version 440 core
#include "size.glsl"

layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec4 vColorTime;
layout (location = 2) in float vTimes[NUM_CAMERAS];
layout (location = 2 + NUM_CAMERAS) in vec4 vNormRad;

out vec4 vPosition0;
out vec4 vColorTime0;
out vec4 vNormRad0;
flat out int id;

void main()
{
    vPosition0 = vPosition;
    vColorTime0 = vColorTime;
    vNormRad0 = vNormRad;
    id = gl_VertexID;    
}