/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#version 440 core

layout(points) in;
layout(points, max_vertices = 1) out;

in vec4 vPosition0[];
in vec4 vColorTime0[];
in vec4 vNormRad0[];
flat in int id[];

out vec4 vData;

uniform int sampleRate;

void main() 
{
    if(id[0] % sampleRate == 0 /*&& vColorTime0[0].w != -3*/)//3500 2500 10000
    {
        vData.xyz = vPosition0[0].xyz;
        vData.w = vColorTime0[0].z;
        EmitVertex();
        EndPrimitive(); 
    }
}
