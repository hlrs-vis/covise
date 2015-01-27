#ifndef __K_Coloring_cu__
#define __K_Coloring_cu__

#include "cutil_math.h"

#include "K_Coloring.cuh"

//from http://www.cs.rit.edu/~ncs/color/t_convert.html
//The hue value H runs from 0 to 360º. 
//The saturation S is the degree of strength or purity and is from 0 to 1. 
//Purity is how much white is added to the color, so S=1 makes the purest color (no white). 
//Brightness V also ranges from 0 to 1, where 0 is the black.
__device__ float3 HSVtoRGB(float h, float s, float v )
{
	float r=0,g=0,b=0;
	int i;
	float f, p, q, t;
	if( s == 0 ) {
		// achromatic (grey)
		r = g = b = v;
		return make_float3(r,g,b);
	}
	h /= 60;			// sector 0 to 5
	i = floor( h );
	f = h - i;			// factorial part of h
	p = v * ( 1.0f - s );
	q = v * ( 1.0f - s * f );
	t = v * ( 1.0f - s * ( 1.0f - f ) );
	switch( i ) {
		case 0:
			r = v;	g = t;	b = p;	
			break;
		case 1:
			r = q;	g = v;	b = p;
			break;
		case 2:
			r = p;	g = v;	b = t;
			break;
		case 3:
			r = p;	g = q;	b = v;
			break;
		case 4:
			r = t;	g = p;	b = v;
			break;
		default:		// case 5:
			r = v;	g = p;	b = q;
			break;
	}

	return make_float3(r,g,b);
}


__device__ float3 calculateColor(ColoringGradient coloringGradient, float colorScalar)
{
	float3 color = make_float3(0,0,0);
	switch(coloringGradient)
	{
	case White:
		// completely white
		{
			color = make_float3(1,1,1);
		}
		break;
	case Blackish:
		// acromatic gradient with V from 0 to 0.5
		{
			float h = colorScalar*0.5;
			color = make_float3(h,h,h);
		}
		break;
	case BlackToCyan:
		{
			color = make_float3(0,colorScalar,colorScalar);
		}
		break;
	case BlueToWhite:
		// blue to white gradient
		{
			color = make_float3(1-colorScalar, 1-0.5f*colorScalar, 1);
		}
		break;
	case HSVBlueToRed:
		// hsv gradient from blue to red (0 to 245 degrees in hue)
		{
			float h = clamp((1-colorScalar)*245.0f,0.0f,245.0f);
			color = HSVtoRGB(h,0.5f,1);
		}
		break;
	}
	return color;
}


static __device__ float3 CalculateColor(ColoringGradient coloringGradient, SPHColoringSource coloringSource, float3 vnext, float pressure, float3 force)
{
	float3 color = make_float3(0);
	switch(coloringSource)
	{
	case Velocity:
		// color given by velocity
		{
			float colorScalar = fabs(vnext.x)+fabs(vnext.y)+fabs(vnext.z) / 11000.0;
			colorScalar = clamp(colorScalar, 0.0f, 1.0f);
			color =  calculateColor(coloringGradient, colorScalar);
		}
		break;
	case Pressure:
		// color given by pressure
		{
			float colorScalar = clamp(( (pressure - cFluidParams.rest_pressure)/ 400.0), 0.1f, 1.0f);
			color =  calculateColor(coloringGradient, colorScalar); 
		}
		break;
	case Force:
		// color given by force
		{
			if(coloringGradient == Direct)
			{
				//color = clamp(make_float3(0.5)+(force/80.0f),make_float3(0),make_float3(1)); 
				color = clamp((fabs(force)/80.0f),make_float3(0),make_float3(1)); 
			}
			else
			{
				force /= 80.0f;
				float colorScalar = clamp((force.x+force.y+force.z)/3.0f,0.1f,1.0f);
				color =  calculateColor(coloringGradient, colorScalar); 
			}
		}
		break;
	}
	return color;
}


#endif