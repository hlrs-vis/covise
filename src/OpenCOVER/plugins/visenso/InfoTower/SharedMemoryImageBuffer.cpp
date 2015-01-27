/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SharedMemoryImageBuffer.h"

#include <windows.h>
#include <GL/gl.h>
#include <iostream>

using namespace std;
using namespace boost;
using namespace boost::interprocess;

SharedMemoryImageBuffer::SharedMemoryImageBuffer(Role role)
    : m_Role(role)
    , m_Filename("InfoTowerImageBuffer")
    , m_ImageBuffer(NULL)
{
    if (m_Role == SharedMemoryImageBuffer::Producer)
    {
        // 		m_remover.reset( new shm_remove(m_Filename));
    }
    else
    {
        bool tryAgain = true;
        while (tryAgain)
        {
            try
            {
                shm_object.reset(new windows_shared_memory(open_only, m_Filename.c_str(), read_only));
                tryAgain = false;
            }
            catch (std::exception &e)
            {
                cerr << "Error accessing shared memory file. Error message was: \"" << e.what() << "\". Trying again in " << m_retryInSeconds << " seconds. " << endl;
                Sleep(m_retryInSeconds * 1000);
                tryAgain = true;
            }
        }
        InitConsumer();
    }
}

SharedMemoryImageBuffer::~SharedMemoryImageBuffer(void)
{
}

int SharedMemoryImageBuffer::GetBufferSize() const
{
    return GetBufferSize(m_ImageWidth, m_ImageHeight, m_ImageFormat);
}

int SharedMemoryImageBuffer::GetBufferSize(int width, int height, unsigned int format) const
{
    switch (format)
    {
    case GL_RGB:
    case GL_BGR_EXT:
        return 3 * width * height;
    case GL_RGBA:
        return 4 * width * height;
    }
    return 0;
}

void
SharedMemoryImageBuffer::SetImageSize(int width, int height, GLenum format)
{
    const int buffer_size = GetBufferSize(width, height, format);
    shm_object.reset(new windows_shared_memory(create_only, m_Filename.c_str(), read_write, sizeof(ImageBuffer) + buffer_size * 2));
    shm_region.reset(new mapped_region(*shm_object, read_write));
    m_ImageBuffer = static_cast<ImageBuffer *>(shm_region->get_address());
    m_ImageBuffer->width = width;
    m_ImageBuffer->height = height;
    m_ImageBuffer->format = format;
    m_ImageBuffer->front_index = 0;
    m_ImageBuffer->back_index = 1;

    m_ImageWidth = m_ImageBuffer->width;
    m_ImageHeight = m_ImageBuffer->height;
    m_ImageFormat = m_ImageBuffer->format;
}

void
SharedMemoryImageBuffer::InitConsumer()
{
    try
    {
        shm_region.reset(new mapped_region(*shm_object, read_only));
    }
    catch (std::exception &e)
    {
        cerr << "shm_region.reset Caught exeption: " << e.what() << endl;
    }
    m_ImageBuffer = static_cast<ImageBuffer *>(shm_region->get_address());
    m_ImageWidth = m_ImageBuffer->width;
    m_ImageHeight = m_ImageBuffer->height;
    m_ImageFormat = m_ImageBuffer->format;
}

void *
SharedMemoryImageBuffer::GetFrontImage() const
{
    const int buffer_size = GetBufferSize(m_ImageWidth, m_ImageHeight, m_ImageFormat);
    return reinterpret_cast<unsigned char *>(&(m_ImageBuffer->buffer_start)) + m_ImageBuffer->front_index * buffer_size;
}

void *
SharedMemoryImageBuffer::GetBackImage() const
{
    const int buffer_size = GetBufferSize(m_ImageWidth, m_ImageHeight, m_ImageFormat);
    void *back_buffer = reinterpret_cast<unsigned char *>(&(m_ImageBuffer->buffer_start)) + m_ImageBuffer->back_index * buffer_size;
    return reinterpret_cast<unsigned char *>(&(m_ImageBuffer->buffer_start)) + m_ImageBuffer->back_index * buffer_size;
}

void
SharedMemoryImageBuffer::SwapBuffers()
{
    int oldFront = m_ImageBuffer->front_index;
    m_ImageBuffer->front_index = m_ImageBuffer->back_index;
    m_ImageBuffer->back_index = oldFront;
}
