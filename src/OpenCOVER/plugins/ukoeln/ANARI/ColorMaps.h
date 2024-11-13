// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <anari/anari_cpp/ext/glm.h>
#include <map>
#include <vector>
#include <string>

namespace util {

  /*! convert PNG rgb image into a color map; if PNG doesn't have an
      alpha channel, alpha channel will be set to constant 1.f */
  struct ColorMap : public std::vector<glm::vec4>
  {
    static ColorMap fromPNG(const uint8_t asPNG[],
                            size_t numBytes);
  
    /*! returns re-sampled version of this same map, with numSamples
        samples */
    ColorMap resampledTo(int numSamples) const;
  };



  
  /*! class that abstracts a library of different, predefined color
      maps that are embedded in this library */
  struct ColorMapLibrary
  {
    ColorMapLibrary();
    
    /*! return a color map library of hardcoded, embedded color
      maps */
    void loadDefaults(int numSamplesPerMap=128);

    /*! adds a color map from a png file to the library */
    void addColorMap(const ColorMap &cm,
                     std::string name,
                     int numSamples=128);
    
    /*! returns a std::vector with all the names of color maps known
      in this library */
    std::vector<std::string> getNames() const;

    /*! find colro map with given name, and reutrn pointer to it; or
        null if this does not exist */
    const ColorMap *find(const std::string &name) const;

    /*! return map with given index */
    const ColorMap &getMap(const int mapID) const;
    
    /*! return _name_ of map with given index */
    const std::string &getMapName(const int mapID) const;

  private:
    
    /*! list of color maps stored in this library */
    std::vector<std::pair<std::string,ColorMap>> knownMaps;
  };

}

