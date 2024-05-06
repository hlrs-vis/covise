// Copyright 2023, The Khronos Group Inc.
//
// SPDX-License-Identifier: Apache-2.0

// OpenXR Tutorial for Khronos Group

#pragma once
#ifdef _MSC_VER
#define NOMINMAX
#include <direct.h>
#include <windows.h>
#ifndef _MAX_PATH
#define _MAX_PATH 500
#endif
#include <time.h>

#include <cerrno>
#include <fstream>
#include <iostream>
#include <sstream>

#ifndef _MSC_VER
#define __stdcall
#endif
typedef void(__stdcall *DebugOutputCallback)(const char *);

class vsBufferedStringStreamBuf : public std::streambuf {
public:
    vsBufferedStringStreamBuf(int bufferSize) {
        if (bufferSize) {
            char *ptr = new char[bufferSize];
            setp(ptr, ptr + bufferSize);
        } else
            setp(0, 0);
    }
    virtual ~vsBufferedStringStreamBuf() {
        // sync();
        delete[] pbase();
    }
    virtual void writeString(const std::string &str) = 0;

private:
    int overflow(int c) {
        sync();

        if (c != EOF) {
            if (pbase() == epptr()) {
                std::string temp;
                temp += char(c);
                writeString(temp);
            } else
                sputc((char)c);
        }

        return 0;
    }

    int sync() {
        if (pbase() != pptr()) {
            int len = int(pptr() - pbase());
            std::string temp(pbase(), len);
            writeString(temp);
            setp(pbase(), epptr());
        }
        return 0;
    }
};

class DebugOutput : public vsBufferedStringStreamBuf {
public:
    DebugOutput(size_t bufsize = (size_t)16)
        : vsBufferedStringStreamBuf((int)bufsize), old_cout_buffer(NULL), old_cerr_buffer(NULL) {
        old_cout_buffer = std::cout.rdbuf(this);
        old_cerr_buffer = std::cerr.rdbuf(this);
    }
    virtual ~DebugOutput() {
        std::cout.rdbuf(old_cout_buffer);
        std::cerr.rdbuf(old_cerr_buffer);
    }
    virtual void writeString(const std::string &str) {
        OutputDebugStringA(str.c_str());
    }

protected:
    std::streambuf *old_cout_buffer;
    std::streambuf *old_cerr_buffer;
};

#elif defined(__linux__) && !defined(__ANDROID__)
#include <iostream>
class DebugOutput {
public:
    DebugOutput() {
        std::cout << "Testing cout redirect." << std::endl;
        std::cerr << "Testing cerr redirect." << std::endl;
    }
};
#elif defined(__ANDROID__)
#include <android/log.h>
#include <stdarg.h>

#include <iostream>
class AndroidStreambuf : public std::streambuf {
public:
    enum {
        bufsize = 128
    };  // ... or some other suitable buffer size
    android_LogPriority logPriority;
    AndroidStreambuf(android_LogPriority p = ANDROID_LOG_DEBUG) {
        logPriority = p;
        this->setp(buffer, buffer + bufsize - 1);
    }

private:
    int overflow(int c) {
        if (c == traits_type::eof()) {
            *this->pptr() = traits_type::to_char_type(c);
            this->sbumpc();
        }
        return this->sync() ? traits_type::eof() : traits_type::not_eof(c);
    }
    std::string str;
    int sync() {
        int rc = 0;
        if (this->pbase() != this->pptr()) {
            char writebuf[bufsize + 1];
            memcpy(writebuf, this->pbase(), this->pptr() - this->pbase());
            writebuf[this->pptr() - this->pbase()] = '\0';
            str += writebuf;
            for (size_t pos = 0; pos < str.length(); pos++) {
                if (str[pos] == '\n') {
                    str[pos] = 0;
                    rc = __android_log_write(logPriority, "openxr_tutorial", str.c_str()) > 0;
                    if (str.length() > pos + 1)
                        str = str.substr(pos + 1, str.length() - pos - 1);
                    else
                        str.clear();
                }
            }
            this->setp(buffer, buffer + bufsize - 1);
        }
        return rc;
    }

    char buffer[bufsize];
};
class DebugOutput {
public:
    AndroidStreambuf androidCout;
    AndroidStreambuf androidCerr;

    DebugOutput() 
        : androidCout(), androidCerr(ANDROID_LOG_ERROR) {
        auto *oldout = std::cout.rdbuf(&androidCout);
        auto *olderr = std::cerr.rdbuf(&androidCerr);
        if (oldout != &androidCout) {
            __android_log_write(ANDROID_LOG_DEBUG, "openxr_tutorial", "redirected cout");
        }
        std::cout << "Testing cout redirect." << std::endl;
        if (olderr != &androidCerr) {
            __android_log_write(ANDROID_LOG_WARN, "openxr_tutorial", "redirected cerr");
        }
        std::cerr << "Testing cerr redirect." << std::endl;
    }
};

#else
class DebugOutput {
public:
    DebugOutput() {
    }
};


#endif

#ifdef __ANDROID__
#include <android/log.h>
#include <sstream>

#define XR_TUT_LOG_TAG "openxr_tutorial"
#define XR_TUT_LOG(...) {                                                           \
        std::ostringstream ostr;                                                    \
        ostr<<__VA_ARGS__;                                                          \
        __android_log_write(ANDROID_LOG_DEBUG, XR_TUT_LOG_TAG, ostr.str().c_str()); \
    }
#define XR_TUT_LOG_ERROR(...) {                                                     \
        std::ostringstream ostr;                                                    \
        ostr<<__VA_ARGS__;                                                          \
        __android_log_write(ANDROID_LOG_ERROR, XR_TUT_LOG_TAG, ostr.str().c_str()); \
    }
#else
#include <iostream>

#define XR_TUT_LOG(...) std::cout << __VA_ARGS__ << "\n"
#define XR_TUT_LOG_ERROR(...) std::cerr << __VA_ARGS__ << "\n"
#endif