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
#include "size.glsl"

layout(points) in;
layout(points, max_vertices = 1) out;

in vec4 vPosition[];
in vec4 vColor[];
in float vTimes[][NUM_CAMERAS];
in vec4 vNormRad[];
in float zVal[];

out vec4 vPosition0;
out vec4 vColor0;
out float vTimes0[NUM_CAMERAS];
out vec4 vNormRad0;

void main() 
{
    if(zVal[0] > 0)
    {
        vPosition0 = vPosition[0];
        vColor0 = vColor[0];
        vNormRad0 = vNormRad[0];
        for(int i = 0; i < vTimes[0].length(); i++)
        {
            vTimes0[i] = vTimes[0][i];
        }
        EmitVertex();
        EndPrimitive(); 
    }
}
