/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CvCapture.h"

CvCapture::CvCapture()
{
    cap = NULL;
    running = false;
    videocallback = NULL;
    images.clear();
}

CvCapture::~CvCapture()
{

    if (clean)
    {
        cap->stop();
        delete cap;
    }

    for (size_t i = 0; i < images.size(); i++)
    {
        if (images[i].release_at_exit)
        {
            cvReleaseImage(&(images[i].ipl));
        }
    }
    images.clear();
}

void CvCapture::default_videocallback(IplImage *image)
{
    // TODO: Skip frames if we are too slow? Impossible using OpenCV?
    /*
	static bool semaphore=false;
	if (semaphore) return;
	semaphore = true;
	*/

    if (CvCapture::Instance().videocallback)
    {
        CvCapture::Instance().videocallback(image);
    }
    CvCapture::Instance().ShowVisibleImages();

    //semaphore = false;
}

CvCapture &CvCapture::Instance()
{
    static CvCapture obj;
    return obj;
}

void CvCapture::SetVideoCallback(void (*_videocallback)(IplImage *image))
{
    videocallback = _videocallback;
}

bool CvCapture::StartVideo(Capture *_cap, const char *_wintitle)
{
    bool clean = false;
    cap = _cap;
    if (cap == NULL)
    {
        CaptureFactory::CaptureDeviceVector vec = CaptureFactory::instance()->enumerateDevices();
        if (vec.size() < 1)
            return false;
        cap = CaptureFactory::instance()->createCapture(vec[0]);
        if (!cap->start())
        {
            delete cap;
            return false;
        }
        clean = true;
    }

    return true;
}

size_t CvCapture::SetImage(const char *title, IplImage *ipl, bool release_at_exit /* =false */)
{
    size_t index = GetImageIndex(title);
    if (index == -1)
    {
        // If the title wasn't found create new
        Image i(ipl, title, false, release_at_exit);
        images.push_back(i);
        return (images.size() - 1);
    }
    // If the title was found replace the image
    if (images[index].release_at_exit)
    {
        cvReleaseImage(&(images[index].ipl));
    }
    images[index].ipl = ipl;
    images[index].release_at_exit = release_at_exit;
    return index;
}

IplImage *CvCapture::CreateImage(const char *title, CvSize size, int depth, int channels)
{
    IplImage *ipl = cvCreateImage(size, depth, channels);
    if (!ipl)
        return NULL;
    SetImage(title, ipl, true);
    return ipl;
}

IplImage *CvCapture::CreateImageWithProto(const char *title, IplImage *proto, int depth /* =0 */, int channels /* =0 */)
{
    if (depth == 0)
        depth = proto->depth;
    if (channels == 0)
        channels = proto->nChannels;
    IplImage *ipl = cvCreateImage(cvSize(proto->width, proto->height), depth, channels);
    if (!ipl)
        return NULL;
    ipl->origin = proto->origin;
    SetImage(title, ipl, true);
    return ipl;
}

IplImage *CvCapture::GetImage(size_t index)
{
    if (index < 0)
        return NULL;
    if (index >= images.size())
        return NULL;
    return images[index].ipl;
}

size_t CvCapture::GetImageIndex(const char *title)
{
    std::string s(title);
    for (size_t i = 0; i < images.size(); i++)
    {
        if (s.compare(images[i].title) == 0)
        {
            return i;
        }
    }
    return (size_t)-1;
}

IplImage *CvCapture::GetImage(const char *title)
{
    return GetImage(GetImageIndex(title));
}

bool CvCapture::ToggleImageVisible(size_t index, int flags)
{
    if (index >= images.size())
        return false;
    if (images[index].visible == false)
    {
        images[index].visible = true;
        cvNamedWindow(images[index].title.c_str(), flags);
        return true;
    }
    else
    {
        images[index].visible = false;
        cvDestroyWindow(images[index].title.c_str());
        return false;
    }
}
