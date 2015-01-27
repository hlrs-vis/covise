/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "cuda.h"
#include "timer.h"



namespace ocu {    
    
    

GPUTimer::GPUTimer() 
{
  e_start = new cudaEvent_t;
  e_stop = new cudaEvent_t;

  cudaEventCreate((cudaEvent_t *)e_start);  
  cudaEventCreate((cudaEvent_t *)e_stop); 
}

GPUTimer::~GPUTimer() 
{ 
  cudaEventDestroy(*((cudaEvent_t *)e_start)); 
  cudaEventDestroy(*((cudaEvent_t *)e_stop)); 

  delete (cudaEvent_t *)e_start;
  delete (cudaEvent_t *)e_stop;
}

void GPUTimer::start() { 
  cudaEventRecord(*((cudaEvent_t *)e_start), 0); 
}

void GPUTimer::stop()  { 
  cudaEventRecord(*((cudaEvent_t *)e_stop), 0); 
}

float GPUTimer::elapsed_ms()
{
    cudaEventSynchronize(*((cudaEvent_t *)e_stop));
    float ms;
    cudaEventElapsedTime(&ms, *((cudaEvent_t *)e_start), *((cudaEvent_t *)e_stop));
    return ms;
}
    
} // end namespace

