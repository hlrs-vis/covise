/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <boost/interprocess/windows_shared_memory.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/offset_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/cstdint.hpp>

class SharedMemoryImageBuffer
{
public:
    enum Role
    {
        Consumer,
        Producer
    };

    SharedMemoryImageBuffer(Role role);
    ~SharedMemoryImageBuffer();

    void SetImageSize(int width, int height, unsigned int format);
    int GetImageWidth() const
    {
        return m_ImageWidth;
    }
    int GetImageHeight() const
    {
        return m_ImageHeight;
    }
    unsigned int GetImageFormat() const
    {
        return m_ImageFormat;
    }
    int GetBufferSize() const;

    void *GetFrontImage() const;
    void *GetBackImage() const;

    void SwapBuffers();

private:
    void InitConsumer();
    int GetBufferSize(int width, int height, unsigned int format) const;

    Role m_Role;
    boost::shared_ptr<boost::interprocess::windows_shared_memory> shm_object;
    boost::shared_ptr<boost::interprocess::mapped_region> shm_region;

    int m_ImageWidth;
    int m_ImageHeight;
    unsigned int m_ImageFormat;
    const std::string m_Filename;

    //struct shm_remove
    //   {
    //	shm_remove(std::string filename) : m_Filename(filename) { boost::interprocess::shared_memory_object::remove(m_Filename.c_str()); }
    //        ~shm_remove() { boost::interprocess::shared_memory_object::remove(m_Filename.c_str()); }
    //	 std::string m_Filename;
    //   };
    //
    //boost::shared_ptr<shm_remove> m_remover;

    struct ImageBuffer
    {
        int width;
        int height;
        unsigned int format;
        int front_index;
        int back_index;
        int buffer_start;
    };

    ImageBuffer *m_ImageBuffer;
    static const int m_retryInSeconds = 1;
};
