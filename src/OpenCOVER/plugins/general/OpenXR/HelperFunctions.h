// Copyright 2023, The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

// OpenXR Tutorial for Khronos Group

#pragma once

// C/C++ Headers
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

// Debugbreak
#if defined(_MSC_VER)
#define DEBUG_BREAK __debugbreak()
#else
#include <signal.h>
#define DEBUG_BREAK raise(SIGTRAP)
#endif

inline bool IsStringInVector(std::vector<const char *> list, const char *name) {
    bool found = false;
    for (auto &item : list) {
        if (strcmp(name, item) == 0) {
            found = true;
            break;
        }
    }
    return found;
}

template <typename T>
inline bool BitwiseCheck(const T &value, const T &checkValue) {
    return ((value & checkValue) == checkValue);
}

template <typename T>
T Align(T value, T alignment) {
    return (value + (alignment - 1)) & ~(alignment - 1);
};

inline std::string GetEnv(const std::string &variable) {
    const char *value = std::getenv(variable.c_str());
    // It's invalid to assign nullptr to std::string
    return value != nullptr ? std::string(value) : std::string("");
}

inline void SetEnv(const std::string &variable, const std::string &value) {
#if defined(_MSC_VER)
    _putenv_s(variable.c_str(), value.c_str());
#else
    setenv(variable.c_str(), value.c_str(), 1);
#endif
}

inline std::string ReadTextFile(const std::string &filepath) {
    std::ifstream stream(filepath, std::fstream::in);
    std::string output;
    if (!stream.is_open()) {
        std::cout << "Could not read file " << filepath.c_str() << ". File does not exist." << std::endl;
        return "";
    }
    std::string line;
    while (!stream.eof()) {
        std::getline(stream, line);
        output.append(line + "\n");
    }
    stream.close();
    return output;
}

inline std::vector<char> ReadBinaryFile(const std::string &filepath) {
    std::ifstream stream(filepath, std::fstream::in | std::fstream::binary | std::fstream::ate);
    if (!stream.is_open()) {
        std::cout << "Could not read file " << filepath.c_str() << ". File does not exist." << std::endl;
        return {};
    }
    std::streamoff size = stream.tellg();
    std::vector<char> output(static_cast<size_t>(size));
    stream.seekg(0, std::fstream::beg);
    stream.read(output.data(), size);
    stream.close();
    return output;
}

#if defined(__ANDROID__)
#include <android/asset_manager.h>
inline std::string ReadTextFile(const std::string &filepath, AAssetManager *assetManager) {
    AAsset *file = AAssetManager_open(assetManager, filepath.c_str(), AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    std::string text;
    text.resize(fileLength);
    AAsset_read(file, (void *)text.data(), fileLength);
    AAsset_close(file);
    return text;
}

inline std::vector<char> ReadBinaryFile(const std::string &filepath, AAssetManager *assetManager) {
    AAsset *file = AAssetManager_open(assetManager, filepath.c_str(), AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    std::vector<char> binary(fileLength);
    AAsset_read(file, (void *)binary.data(), fileLength);
    AAsset_close(file);
    return binary;
}
#endif

#ifdef _MSC_VER
#define strncpy(dst, src, count) strcpy_s(dst, count, src);
#endif


#define XR_DOCS_CHAPTER_1_4 0x14

#define XR_DOCS_CHAPTER_2_1 0x21
#define XR_DOCS_CHAPTER_2_2 0x22
#define XR_DOCS_CHAPTER_2_3 0x23

#define XR_DOCS_CHAPTER_3_1 0x31
#define XR_DOCS_CHAPTER_3_2 0x32
#define XR_DOCS_CHAPTER_3_3 0x33

#define XR_DOCS_CHAPTER_4_1 0x41
#define XR_DOCS_CHAPTER_4_2 0x42
#define XR_DOCS_CHAPTER_4_3 0x43
#define XR_DOCS_CHAPTER_4_4 0x44
#define XR_DOCS_CHAPTER_4_5 0x45
#define XR_DOCS_CHAPTER_4_6 0x46

#define XR_DOCS_CHAPTER_5_1 0x51
#define XR_DOCS_CHAPTER_5_2 0x52
#define XR_DOCS_CHAPTER_5_3 0x53
#define XR_DOCS_CHAPTER_5_4 0x54
#define XR_DOCS_CHAPTER_5_5 0x55
#define XR_DOCS_CHAPTER_5_6 0x56
