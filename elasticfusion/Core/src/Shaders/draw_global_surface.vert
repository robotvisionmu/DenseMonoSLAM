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

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 color;
layout (location = 2) in float times[NUM_CAMERAS];
layout (location = 2 + NUM_CAMERAS) in vec4 normal;

uniform mat4 MVP;
uniform float threshold;
uniform int colorType;
uniform int unstable;
uniform int drawWindow;
uniform int time;
uniform int timeIdx;
uniform int timeDelta;

out vec4 vColor;
out float vTimes[NUM_CAMERAS];
out vec4 vPosition;
out vec4 vNormRad;
out mat4 vMVP;
out int vTime;
out int vTimeIdx;
out int colorType0;
out int drawWindow0;
out int timeDelta0;

void main()
{
    if(position.w > threshold || unstable == 1)
    {
        colorType0 = colorType;
        drawWindow0 = drawWindow;
	    vColor = color;
	    vPosition = position;
	    vNormRad = normal;
	    vMVP = MVP;
	    vTime = time;
        vTimeIdx = timeIdx;
	    timeDelta0 = timeDelta;
	    gl_Position = MVP * vec4(position.xyz, 1.0);

        for(int i = 0; i < times.length(); i++)
        {
            vTimes[i] = times[i];
        }
    }
    else
    {
        colorType0 = -1;
    }
}
