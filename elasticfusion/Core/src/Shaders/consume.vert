#version 440 core
#include "size.glsl"

layout (location = 0) in vec4 vPos;
layout (location = 1) in vec4 vCol;
layout (location = 2) in float vTs[NUM_CAMERAS];
layout (location = 2 + NUM_CAMERAS) in vec4 vNormR;

out vec4 vPosition0;
out vec4 vColor0;
out float vTimes0[NUM_CAMERAS];
out vec4 vNormRad0;

uniform mat4 transform;

void main()
{
    //mat3 r = mat3(transform);
    //vec3 t = vec3(transform[0][3], transform[1][3], transform[2][3]);

    vPosition0 = transform * vec4(vPos.xyz, 1);// + t;
    vPosition0.w = vPos.w;
    vColor0 = vCol;
    vNormRad0.w = vNormR.w;
    vNormRad0.xyz = mat3(transform) * vNormR.xyz;

    for(int i = 0; i < vTs.length(); i++)
    {
        vTimes0[i] = vTs[i];
    }
}