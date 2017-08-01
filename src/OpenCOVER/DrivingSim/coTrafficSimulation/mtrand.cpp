/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// mtrand.cpp, see include file mtrand.h for information

#include "mtrand.h"
// non-inline function definitions and static member definitions cannot
// reside in header file because of the risk of multiple declarations

// initialization of static private members

unsigned long MTRand_int32::state[MTRand_int32_N] = { 0x0UL };
int MTRand_int32::p = 0;
bool MTRand_int32::init = false;


void MTRand_int32::gen_state()
{ // generate new state vector
    for (int i = 0; i < (MTRand_int32_N - MTRand_int32_M); ++i)
        state[i] = state[i + MTRand_int32_M] ^ twiddle(state[i], state[i + 1]);
    for (int i = MTRand_int32_N - MTRand_int32_M; i < (MTRand_int32_N - 1); ++i)
        state[i] = state[i + MTRand_int32_M - MTRand_int32_N] ^ twiddle(state[i], state[i + 1]);
    state[MTRand_int32_N - 1] = state[MTRand_int32_M - 1] ^ twiddle(state[MTRand_int32_N - 1], state[0]);
    p = 0; // reset position
}

void MTRand_int32::seed(unsigned long s)
{ // init by 32 bit seed
    state[0] = s & 0xFFFFFFFFUL; // for > 32 bit machines
    for (int i = 1; i < MTRand_int32_N; ++i)
    {
        state[i] = 1812433253UL * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
        // see Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier
        // in the previous versions, MSBs of the seed affect only MSBs of the array state
        // 2002/01/09 modified by Makoto Matsumoto
        state[i] &= 0xFFFFFFFFUL; // for > 32 bit machines
    }
    p = MTRand_int32_N; // force gen_state() to be called for next random number
}

void MTRand_int32::seed(const unsigned long *array, int size)
{ // init by array
    seed(19650218UL);
    int i = 1, j = 0;
    for (int k = ((MTRand_int32_N > size) ? MTRand_int32_N : size); k; --k)
    {
        state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1664525UL))
                   + array[j] + j; // non linear
        state[i] &= 0xFFFFFFFFUL; // for > 32 bit machines
        ++j;
        j %= size;
        if ((++i) == MTRand_int32_N)
        {
            state[0] = state[MTRand_int32_N - 1];
            i = 1;
        }
    }
    for (int k = MTRand_int32_N - 1; k; --k)
    {
        state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 1566083941UL)) - i;
        state[i] &= 0xFFFFFFFFUL; // for > 32 bit machines
        if ((++i) == MTRand_int32_N)
        {
            state[0] = state[MTRand_int32_N - 1];
            i = 1;
        }
    }
    state[0] = 0x80000000UL; // MSB is 1; assuring non-zero initial array
    p = MTRand_int32_N; // force gen_state() to be called for next random number
}
