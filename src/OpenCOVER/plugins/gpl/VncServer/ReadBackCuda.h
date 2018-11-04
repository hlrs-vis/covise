#ifndef COMPRESS_H

#include <sysdep/opengl.h>
#include <string>

struct cudaGraphicsResource;
typedef unsigned char uchar;

class ReadBackError
{
   public:
   ReadBackError(const std::string &msg)
      : message(msg)
   {}
   const std::string &getMessage() const { return message; }
   private:
   std::string message;
};

class ReadBackCuda
{
   public:
      ReadBackCuda();
      ~ReadBackCuda();

      bool readpixelsyuv(GLint x, GLint y, GLint w, GLint pitch, GLint h,
            GLenum format, int ps, GLubyte *bits, GLint buf, int subx, int suby);
      bool readpixels(GLint x, GLint y, GLint w, GLint pitch, GLint h,
            GLenum format, int ps, GLubyte *bits, GLint buf);

   private:
      bool initPbo(size_t sz, size_t subsz);
      GLuint pboName;
      cudaGraphicsResource* imgRes;
      uchar* outImg;
      size_t imgSize;
};
#endif
