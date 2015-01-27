/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*============================================================================*/
/*                                                                            */
/*      Header   :  CoviseWindowCapture.H                                     */
/*                                                                            */
/*      Author   :  Thomas van Reimersdahl                                    */
/*                                                                            */
/*      Date     :  Januar 2003                                               */
/*                                                                            */
/*============================================================================*/
#ifndef __COVISEWINDOWCAPTURE_H
#define __COVISEWINDOWCAPTURE_H

#include <util/coTypes.h>

/*============================================================================*/
/* INCLUDES                                                                   */
/*============================================================================*/
#include <iostream>

/*============================================================================*/
/* MACROS AND DEFINES                                                         */
/*============================================================================*/

/*============================================================================*/
/* FORWARD DECLARATIONS                                                       */
/*============================================================================*/

/*============================================================================*/
/* CLASS DEFINITIONS                                                          */
/*============================================================================*/
class CCoviseWindowCapture
{
public:
    CCoviseWindowCapture();
    virtual ~CCoviseWindowCapture();

    void Init();
    void SetWidth(int iWidth);
    void SetHeight(int iHeight);
    bool Write();

    void SetFileName(std::string strFileName);
    std::string GetFileName();
    void SetCurrentFrame(int iFrameNr);
    void SetCaptureFrameSequence(bool bOn = true);
    bool GetCaptureFrameSequence();
    bool SaveRenderWindow(void *w,
                          int x,
                          int y,
                          int pixels,
                          int scanlines,
                          char *filename);

    void WriteTGAHeader(FILE *ifp, int width, int height, int bits);

    bool GetRenderWindow(void *w, int x, int y, int pixels, int scanlines, unsigned char *image);

protected:
    int m_iCurrentFrame;
    bool m_bCaptureFrameSequence;
    std::string m_strFileSuffix;
    std::string m_strFileName;
    int m_iWindowHeight;
    int m_iWindowWidth;
};

/*============================================================================*/
/* LOCAL VARS AND FUNCS                                                       */
/*============================================================================*/

/*============================================================================*/
/* END OF FILE                                                                */
/*============================================================================*/
#endif
