/* This file is part of COVISE.
   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <cassert>
#include <cstring>
#include <any>
#include <array>
#include <string>

enum class DataType
{
    Bool,
    Int32,
    Float32,
    Float32_Vec4,
    Unknown,
};

inline
std::string toString(DataType type) {
    switch (type) {
    case DataType::Bool:          return "Bool";
    case DataType::Int32:         return "Int32";
    case DataType::Float32:       return "Float32";
    case DataType::Float32_Vec4:  return "Float32_Vec4";
    default:
    case DataType::Unknown:       return "Unknown";
    }
}

inline
size_t sizeInBytes(DataType type) {
    switch (type) {
    case DataType::Bool:          return sizeof(bool);
    case DataType::Int32:         return sizeof(int32_t);
    case DataType::Float32:       return sizeof(float);
    case DataType::Float32_Vec4:  return sizeof(float[4]);
    default:
    case DataType::Unknown:       return 0;
    }
}

struct Param
{
    std::string name;
    DataType    type = DataType::Unknown;
    std::any    value;

    template <typename T>
    T as() const
    { return std::any_cast<T>(value); }

    void serialize(uint8_t *bytesAllocated) {
        if (type == DataType::Bool) {
            auto val = as<bool>();
            std::memcpy(bytesAllocated, &val, sizeof(val));
        } else if (type == DataType::Int32) {
            auto val = as<int32_t>();
            std::memcpy(bytesAllocated, &val, sizeof(val));
        } else if (type == DataType::Float32) {
            auto val = as<float>();
            std::memcpy(bytesAllocated, &val, sizeof(val));
        } else if (type == DataType::Float32_Vec4) {
            auto val = as<std::array<float,4>>();
            std::memcpy(bytesAllocated, &val, sizeof(val));
        } else {
            assert(0);
        }
    }

    void unserialize(const uint8_t *bytes)
    {
        if (type == DataType::Bool) {
            bool val;
            std::memcpy(&val, bytes, sizeof(val));
            value = val;
        } else if (type == DataType::Int32) {
            int32_t val;
            std::memcpy(&val, bytes, sizeof(val));
            value = val;
        } else if (type == DataType::Float32) {
            float val;
            std::memcpy(&val, bytes, sizeof(val));
            value = val;
        } else if (type == DataType::Float32_Vec4) {
            std::array<float,4> val;
            std::memcpy(&val, bytes, sizeof(val));
            value = val;
        } else {
            assert(0);
        }
    }
};
