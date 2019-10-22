// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef GLM_MATRIXINTERPOLATION_H
#define GLM_MATRIXINTERPOLATION_H


#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>


glm::mat4 interpolate(const glm::mat4& a, const glm::mat4& b, float t);


#endif // #ifndef GLM_MATRIXINTERPOLATION_H
