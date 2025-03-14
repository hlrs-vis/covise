#pragma once
#include <vsg/maths/mat4.h>
#include <vsg/maths/vec3.h>



namespace vive
{
    template<typename T>
    constexpr vsg::t_vec3<T> getTrans(const vsg::t_mat4<T> &mat)
    {
        return vsg::t_vec3<T>(mat[3][0], mat[3][1], mat[3][2]);
    }
    template<typename T>
    constexpr void setTrans(vsg::t_mat4<T>& mat, vsg::t_vec3<T>vec)
    {
        mat[3][0] = vec[0];
        mat[3][1] = vec[1];
        mat[3][2] = vec[2];
    }
    template<typename T>
    constexpr vsg::t_mat4<T> makeEulerMat(T h, T p, T r)
    {
        vsg::t_mat4<T> m;
        double sr, sp, sh, cr, cp, ch;    
            sr = sin(r / 180.0 * M_PI);       
            sp = sin(p / 180.0 * M_PI);       
            sh = sin(h / 180.0 * M_PI);       
            cr = cos(r / 180.0 * M_PI);       
            cp = cos(p / 180.0 * M_PI);       
            ch = cos(h / 180.0 * M_PI);       
            m[0][0] = ch * cr - sh * sr * sp;
            m[0][1] = cr * sh + ch * sr * sp;
            m[0][2] = -sr * cp;
            m[0][3] = 0;
            m[1][0] = -sh * cp;
            m[1][1] = ch * cp;
            m[1][2] = sp;
            m[1][3] = 0;
            m[2][0] = sp * cr * sh + sr * ch;
            m[2][1] = sr * sh - sp * cr * ch;
            m[2][2] = cp * cr;
            m[2][3] = 0;
            m[3][0] = 0;
            m[3][1] = 0;
            m[3][2] = 0;
            m[3][3] = 1;
        return m;
    }
    template<typename T>
    constexpr vsg::t_mat4<T> makeEulerMat(vsg::t_vec3<T> hpr)
    {
        return(makeEulerMat(hpr[0], hpr[1], hpr[2]));
    }

}
