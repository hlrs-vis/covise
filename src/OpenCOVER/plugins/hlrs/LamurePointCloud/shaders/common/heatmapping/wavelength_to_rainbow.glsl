// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

// "private" interfaces for color ramping
// --------------------------
const float GAMMA        = 0.80;
const float INTENSITY_MAX = 255.0;
	
float round(float d){
  return floor(d + 0.5);
}
	
float adjust(in float color, in float factor){
  if (color == 0.0){
    return 0.0;
  }
  else{
    float res = round(INTENSITY_MAX * pow(color * factor, GAMMA));
    return min(255.0, max(0.0, res));
  }
}

vec3 wavelength_to_RGB(in float wavelength){
  float Blue;
  float factor;
  float Green;
  float Red;
  if(380.0 <= wavelength && wavelength <= 440.0){
    Red   = -(wavelength - 440.0) / (440.0 - 380.0);
    Green = 0.0;
    Blue  = 1.0;
  }
  else if(440.0 < wavelength && wavelength <= 490.0){
    Red   = 0.0;
    Green = (wavelength - 440.0) / (490.0 - 440.0);
    Blue  = 1.0;
  }
  else if(490.0 < wavelength && wavelength <= 510.0){
    Red   = 0.0;
    Green = 1.0;
    Blue  = -(wavelength - 510.0) / (510.0 - 490.0);
  }
  else if(510.0 < wavelength && wavelength <= 580.0){
    Red   = (wavelength - 510.0) / (580.0 - 510.0);
    Green = 1.0;
    Blue  = 0.0;
  }
  else if(580.0 < wavelength && wavelength <= 645.0){		
    Red   = 1.0;
    Green = -(wavelength - 645.0) / (645.0 - 580.0);
    Blue  = 0.0;
  }
  else if(645.0 < wavelength && wavelength <= 780.0){
    Red   = 1.0;
    Green = 0.0;
    Blue  = 0.0;
  }
  else{
    Red   = 0.0;
    Green = 0.0;
    Blue  = 0.0;
  }
  
  
  if(380.0 <= wavelength && wavelength <= 420.0){
    factor = 0.3 + 0.7*(wavelength - 380.0) / (420.0 - 380.0);
  }
  else if(420.0 < wavelength && wavelength <= 701.0){
    factor = 1.0;
  }
  else if(701.0 < wavelength && wavelength <= 780.0){
    factor = 0.3 + 0.7*(780.0 - wavelength) / (780.0 - 701.0);
  }
  else{
    factor = 0.0;
  }
  float R = adjust(Red,   factor);
  float G = adjust(Green, factor);
  float B = adjust(Blue,  factor);
  return vec3(R/255.0,G/255.0,B/255.0);
}
	
	
	
	
float get_wavelength_from_data_point(float value, float min_value, float max_value){
  float min_visible_wavelength = 380.0;//350.0;
  float max_visible_wavelength = 780.0;//650.0;
  //Convert data value in the range of MinValues..MaxValues to the 
  //range 350..780
  return (value - min_value) / (max_value-min_value) * (max_visible_wavelength - min_visible_wavelength) + min_visible_wavelength;
}	
	
	
vec3 data_value_to_rainbow(float value, float min_value, float max_value) {
  float wavelength = get_wavelength_from_data_point(value, min_value, max_value);
  return wavelength_to_RGB(wavelength);	  
}
// ------------------
