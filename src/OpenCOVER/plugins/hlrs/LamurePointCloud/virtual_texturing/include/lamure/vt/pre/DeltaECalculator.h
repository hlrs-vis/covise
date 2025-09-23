// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef TILE_PROVIDER_DELTAECALCULATOR_H
#define TILE_PROVIDER_DELTAECALCULATOR_H

//#define DELTA_E_CALCULATOR_LOG_PROGRESS

#include <chrono>
#include <iomanip>
#include <iostream>
#include <lamure/vt/pre/AtlasFile.h>
#include <lamure/vt/pre/OffsetIndex.h>

namespace vt {
    namespace pre {
    class VIRTUAL_TEXTURING_DLL DeltaECalculator : public AtlasFile
    {
        public:
            explicit DeltaECalculator(const char *fileName);
            void calculate(size_t maxMemory);
        };
    }
}

#endif //TILE_PROVIDER_DELTAECALCULATOR_H
