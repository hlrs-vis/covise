/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#ifdef YAC
#include <api/coAPI.h>
#else
#include <covise/covise.h>
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include "CoviseWindowCapture.h"

/*============================================================================*/
/*  CONSTRUCTORS / DESTRUCTOR                                                 */
/*============================================================================*/
CCoviseWindowCapture::CCoviseWindowCapture()
{
    //	m_pCurWindow = NULL;
}

CCoviseWindowCapture::~CCoviseWindowCapture()
{
}

/*============================================================================*/
/* IMPLEMENTATION                                                             */
/*============================================================================*/
/*============================================================================*/
/*                                                                            */
/*  NAME    : Init                                                            */
/*                                                                            */
/*============================================================================*/
void CCoviseWindowCapture::Init()
{
    m_strFileName = std::string("ImageQtRenderer");
    m_strFileSuffix = std::string(".tga");
    m_bCaptureFrameSequence = true;
    m_iCurrentFrame = 0;
    m_iWindowWidth = 600;
    m_iWindowHeight = 400;
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : Write                                                           */
/*                                                                            */
/*============================================================================*/
bool CCoviseWindowCapture::Write()
{
    std::string strTemp;

    if (m_bCaptureFrameSequence == true)
    {
        m_iCurrentFrame++;
        strTemp = m_strFileName;
        char aTemp[10];
        sprintf(aTemp, "%03i", m_iCurrentFrame);
        strTemp.append(aTemp);
        strTemp.append(m_strFileSuffix);
    }
    else
    {
        strTemp = m_strFileName;
        strTemp.append(m_strFileSuffix);
    }
    return this->SaveRenderWindow(NULL, //m_pCurWindow,
                                  0,
                                  0,
                                  m_iWindowWidth,
                                  m_iWindowHeight,
                                  (char *)strTemp.c_str())
               == true
               ? true
               : false;
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : WriteTGAHeader                                                  */
/*                                                                            */
/*============================================================================*/
void CCoviseWindowCapture::WriteTGAHeader(
    FILE *ifp,
    int width,
    int height,
    int bits)
{

    putc(0, ifp); /* IDLength */
    putc(0, ifp); /* CoMapType   */
    putc(2, ifp); /* ImgType     */
    putc(0, ifp); /* Index_lo */
    putc(0, ifp); /* Index_hi */
    putc(0, ifp); /* Length_lo   */
    putc(0, ifp); /* Length_hi   */
    putc(0, ifp); /* CoSize     */
    putc(0, ifp); /* X_org_lo */
    putc(0, ifp); /* X_org_hi */
    putc(0, ifp); /* Y_org_lo */
    putc(0, ifp); /* Y_org_hi */
    putc(width & 255, ifp); /* width_lo   */
    putc(width >> 8, ifp); /* width_hi   */
    putc(height & 255, ifp); /* height_lo   */
    putc(height >> 8, ifp); /* height_hi   */
    putc(bits, ifp); /* PixelSize   */
    putc(0x0, ifp); /* flags       */
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : GetRenderWindow                                               */
/*                                                                            */
/*============================================================================*/
bool CCoviseWindowCapture::GetRenderWindow(void *w,
                                           int x, int y, int pixels, int scanlines, unsigned char *image)
{
    /*#if MPOGL
       unsigned char *simage;
   #endif
      static int lastxsize=0, lastysize=0;
   */
    (void)w;
    if (image == NULL)
        return false;
    /*    if((x > lastxsize) || (y >  lastysize)) return false;
       if(((x+pixels)>lastxsize) || ((y+scanlines)>lastysize))
         return false;
   */

    //    _WTbind_context(w);
    glReadBuffer(GL_FRONT);
    glReadPixels(x, y, pixels, scanlines, GL_RGBA,
                 GL_UNSIGNED_BYTE, image);
    glReadBuffer(GL_BACK);

    return true;
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : SaveRenderWindow                                              */
/*                                                                            */
/*============================================================================*/
bool CCoviseWindowCapture::SaveRenderWindow(void *w,
                                            int x,
                                            int y,
                                            int pixels,
                                            int scanlines,
                                            char *filename)
{
    FILE *fout = NULL;
    unsigned char *image;
    unsigned char *scanline;
    unsigned char *ip, *op;

    // does the window exist?
    //    if((w==NULL) || (filename==NULL)) return false;

    /* 
       if((x >= w->lastxsize) || (y >= w->lastysize)) return false;
       if(((x+pixels)>w->lastxsize) || ((y+scanlines)>w->lastysize))
         return false;
   */

    // check if file can be written
    if (fout == NULL)
        fout = fopen(filename, "wb");
    if (fout == NULL)
        return false;

    // allocate memory for image
    image = new unsigned char[pixels * scanlines * 4];
    if (image == NULL)
        return false;

    // get the window image
    if (!this->GetRenderWindow(w, x, y, pixels, scanlines, image))
    {
        // WTfree(image);
        delete image;
        fclose(fout);
        return false;
    }

    // write the TGA File
    // first, write the header
    WriteTGAHeader(fout, pixels, scanlines, 24);

    // second, write the body
    scanline = new unsigned char[pixels * 3];
    for (y = 0; y < scanlines; y++)
    {
        ip = image + pixels * y * 4;
        op = scanline;
        for (x = 0; x < pixels; x++)
        {
            op[0] = ip[2];
            op[1] = ip[1];
            op[2] = ip[0];
            op += 3;
            ip += 4;
        }
        size_t retval;
        retval = fwrite(scanline, 3, pixels, fout);
        if (((int)retval) != pixels)
        {
            std::cerr << "CCoviseWindowCapture::SaveRenderWindow: fwrite failed" << std::endl;
            return false;
        }
    }
    fclose(fout);

    // clean up allocated memory
    delete image;
    delete scanline;

    return true;
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : SetFileName                                                     */
/*                                                                            */
/*============================================================================*/
void CCoviseWindowCapture::SetFileName(std::string strFileName)
{
    m_strFileName = strFileName;
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : GetFileName                                                     */
/*                                                                            */
/*============================================================================*/
std::string CCoviseWindowCapture::GetFileName()
{
    return m_strFileName;
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : SetCurrentFrame                                                 */
/*                                                                            */
/*============================================================================*/
void CCoviseWindowCapture::SetCurrentFrame(int iFrameNr)
{
    m_iCurrentFrame = iFrameNr;
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : SetCaptureFrameSequence                                         */
/*                                                                            */
/*============================================================================*/
void CCoviseWindowCapture::SetCaptureFrameSequence(bool bOn)
{
    m_bCaptureFrameSequence = bOn;
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : GetCaptureFrameSequence                                         */
/*                                                                            */
/*============================================================================*/
bool CCoviseWindowCapture::GetCaptureFrameSequence()
{
    return m_bCaptureFrameSequence;
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : SetWidth                                                        */
/*                                                                            */
/*============================================================================*/
void CCoviseWindowCapture::SetWidth(int iWidth)
{
    m_iWindowWidth = iWidth;
}

/*============================================================================*/
/*                                                                            */
/*  NAME    : SetHeight                                                       */
/*                                                                            */
/*============================================================================*/
void CCoviseWindowCapture::SetHeight(int iHeight)
{
    m_iWindowHeight = iHeight;
}

/*============================================================================*/
/* LOCAL VARS AND FUNCS                                                       */
/*============================================================================*/

/*============================================================================*/
/* END OF FILE "CoviseWindowCapture.cpp"                                    */
/*============================================================================*/
