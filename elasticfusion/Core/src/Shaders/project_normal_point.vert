#version 440 core

layout (location = 0) in vec2 texcoord;

out vec4 vPosition;
out vec4 vNormal;

uniform sampler2D drSampler;
uniform sampler2D drfSampler;
uniform sampler2D vertSampler;
uniform sampler2D normSampler;

uniform float cols;
uniform float rows;
uniform float scale;
uniform mat4 pose;
uniform vec4 cam;
uniform float maxDepth;
uniform int sampleRate;

#include "geometry.glsl"

void main()
{
    float x = texcoord.x * cols;
    float y = texcoord.y * rows;

    vec3 vPosLocal = textureLod(vertSampler, texcoord, 0).xyz;//getVertex(texcoord.xy, x, y, cam, drSampler);
    vPosition = pose * vec4(vPosLocal, 1);

    //vec3 vPosition_f = getVertex(texcoord.xy, x, y, cam, drfSampler);

    vec3 vNormLocal = textureLod(normSampler, texcoord, 0).xyz;//getNormal(vPosition_f, texcoord.xy, x, y, cam, drfSampler);
    vNormal = vec4(mat3(pose) * vNormLocal, 1);

    if(vPosLocal.z <= 0 || vPosLocal.z >= maxDepth || gl_VertexID % sampleRate != 0)
    {
        //signal to geom shader that this vertex shouldn't
        // be copied to transform feedback
        vPosition.z = -1;
    }
}