// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include "MatrixInterpolation.hpp"


  glm::mat4 interpolate(const glm::mat4& ma, const glm::mat4& mb, float t){

    glm::quat firstQuat  = glm::quat_cast(ma);
    glm::quat secondQuat = glm::quat_cast(mb);
    glm::quat finalQuat  = glm::slerp(firstQuat, secondQuat, t);

    glm::mat4 r = glm::mat4_cast(finalQuat);

    glm::vec4 transformComp1 = glm::vec4(ma[3][0],ma[3][1],ma[3][2],ma[3][3]);
    glm::vec4 transformComp2 = glm::vec4(mb[3][0],mb[3][1],mb[3][2],mb[3][3]);
    
    glm::vec4 finalTrans = (1.0f-t)*transformComp1 + t*transformComp2;
    r[3][0] = finalTrans.x;
    r[3][1] = finalTrans.y;
    r[3][2] = finalTrans.z;
    r[3][3] = finalTrans.w;

    return r;

  }

