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

layout (location = 0) in vec4 vPosition;
layout (location = 1) in vec4 vColor;
layout (location = 2) in float vTimes[NUM_CAMERAS];
layout (location = 2 + NUM_CAMERAS) in vec4 vNormRad;

out vec4 vPosition0;
out vec4 vColor0;
out float vTimes0[NUM_CAMERAS];
out vec4 vNormRad0;

uniform mat4 t_inv;

void main()
{
    vPosition0 = vPosition;
    vColor0 = vColor;
    vColor0.y = 0; //Unused
    vColor0.z = 1; //This sets the vertex's initialisation time
    vNormRad0 = vNormRad;

    //initialise time windows
    for(int i = 0; i < vTimes.length(); i++)
    {
        vTimes0[i] = vTimes[i];
    }
}
