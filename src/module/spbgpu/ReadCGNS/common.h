/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * common.h
 *
 * Some common declarations
 *
 *  Created on: 25.01.2010
 *      Author: Vlad
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <string>

//"\e[31;4m"
const char cout_red[] = "\x1b[31m"; // errors and important warnings
const char cout_green[] = "\x1b[32m";
const char cout_magenta[] = "\x1b[35m"; // warnings
const char cout_cyan[] = "\x1b[36m";
const char cout_underln[] = "\x1b[4m";
const char cout_norm[] = "\x1b[0m";

enum
{
    T_GRID,
    T_VEC3,
    T_FLOAT //parameters for DO creation
};

// parameter structure
struct params
{
    bool b_load_2d;
    bool b_use_string;
    std::string sections_string;

    int param_vx, param_vy, param_vz;

    int param_f[4];
};


#endif /* COMMON_H_ */
