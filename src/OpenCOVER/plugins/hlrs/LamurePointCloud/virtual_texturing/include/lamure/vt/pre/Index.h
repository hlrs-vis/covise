// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_INDEX_H
#define LAMURE_INDEX_H

#include <cstdint>
#include <cstddef>
#include <fstream>

namespace vt {
    namespace pre {
        template<typename val_type>
        class Index {
        protected:
            size_t _size;
            val_type *_data;

            void _switchEndian(uint8_t *data, size_t size){
                size_t valTypeSize = sizeof(val_type);
                size_t byteSize = size * valTypeSize;
                size_t currentI = 0;

                for(size_t i = 0; i < size; ++i){
                    val_type val = ((val_type*)data)[i];

                    for(size_t n = 0; n < valTypeSize; ++n){
                        data[currentI] = (val >> (n << 3)) & 0xff;
                        ++currentI;
                    }
                }
            }

            bool _isBE(){
                uint16_t temp = 1;

                return ((uint8_t*)&temp)[0] == 0;
            }

        public:
            explicit Index<val_type>(size_t size){
                _size = size;
                _data = new val_type[size];
            }

            val_type getValue(uint64_t idx){
                return _data[idx];
            }

            void writeToFile(std::fstream &file){
                bool isBE = _isBE();

                if(isBE){
                    _switchEndian((uint8_t*)_data, _size);
                }

                file.write((char*)_data, _size * sizeof(val_type));

                if(isBE){
                    _switchEndian((uint8_t*)_data, _size);
                }
            }

            void readFromFile(std::fstream &file){
                file.read((char*)_data, _size * sizeof(val_type));

                if(_isBE()){
                    _switchEndian((uint8_t*)_data, _size);
                }
            }

            void readFromFile(std::ifstream &file){
                file.read((char*)_data, _size * sizeof(val_type));

                if(_isBE()){
                    _switchEndian((uint8_t*)_data, _size);
                }
            }

            size_t getByteSize(){
                return sizeof(val_type) * _size;
            }
        };
    }
}

#endif //LAMURE_INDEX_H
