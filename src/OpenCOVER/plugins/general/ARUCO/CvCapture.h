/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CVTESTBED_H
#define CVTESTBED_H

#include "Alvar.h"
#include <vector>
#include <string>
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "platform/CaptureFactory.h"

using namespace alvar;

/**
 * \brief \e CvCapture is a class for making quick OpenCV test applications
 * \author ttehop
 *
 * Usage:
 * \code
 * #include "CvCapture.h"
 * 
 * void videocallback(IplImage *image)
 * {
 *     static IplImage *img_gray=NULL;
 *
 *     assert(image);
 *     if (img_gray == NULL) {
 *         // Following image is toggled visible using key '0'
 *         img_gray = CvCapture::Instance().CreateImageWithProto("img_gray", image, 0, 1);
 *     }
 *     cvCvtColor(image, img_gray, CV_RGB2GRAY);
 *     // TODO: Do your image operations
 * }
 *
 * void main() {
 *	   CvCapture::Instance().SetVideoCallback(videocallback); // Set video callback
 *	   CvCapture::Instance().StartVideo("movie.avi"); // Video from avi
 *	   // CvCapture::Instance().StartVideo(0) // Video from camera 0
 * }
 * \endcode
 *
 * In addition to handling video input from avi or from camera
 * \e CvCapture has also functions for creating and showing
 * (and automatically releasing) IplImage's. 
 *
 * The \e CvCapture is made into a simple Meyers singleton:
 * \code
 * class Singleton {
 * public:
 *     static Singleton& Instance() {
 *         static Singleton obj;
 *         return obj;
 *     }
 * private:
 *     Singleton();
 *     Singleton(const Singleton&);
 *     Singleton& operator=(const Singleton&);
 *     ~Singleton();
 * }
 * \endcode
 *
 * The only instance of the class is accessed using public 
 * Instance()-interface. All possible constructors and destructors
 * are hidden. If more complex singleton approach is needed 
 * refer to Loki library or the book "Modern C++ Design".
 */
class CvCapture
{
protected:
    bool clean;
    Capture *cap;
    /** \brief Hidden constructor for Singleton */
    CvCapture();
    /** \brief Hidden copy constructor for Singleton */
    CvCapture(const CvCapture &);
    /** \brief Hidden copy operator for Singleton */
    CvCapture &operator=(const CvCapture &);
    /** \brief Hidden destructor for Singleton */
    ~CvCapture();
    /** \brief Boolean indicating are we still running. We exit from the \e WaitKeys when this is false. */
    bool running;

    /** \brief Pointer for the user-defined videocallback. */
    void (*videocallback)(IplImage *image);
    /** \brief Pointer for the user-defined KEYcallback. */
    int (*keycallback)(int key);
    /** \brief The window title for the video view. */
    std::string wintitle;
    /** \brief The filename if we are reading an AVI file. */
    std::string filename;
    /** \brief Image structure to store the images internally */
    struct Image
    {
        IplImage *ipl;
        std::string title;
        bool visible;
        bool release_at_exit;
        Image(IplImage *_ipl, std::string _title, bool _visible, bool _release_at_exit)
            : ipl(_ipl)
            , title(_title)
            , visible(_visible)
            , release_at_exit(_release_at_exit)
        {
        }
    };
    /** \brief Vector of images stored internally */
    std::vector<Image> images;

    /** \brief Video callback called for every frame. This calls user-defined videocallback if one exists. */
    static void default_videocallback(IplImage *image);
    /** \brief \e WaitKeys contains the main loop. */
    void WaitKeys();
    /** \brief \e ShowVisibleImages is called from the videocallback. This shows the internally stored images which have the visible-flag on */
    void ShowVisibleImages();

public:
    //CameraDescription camera_description;
    /** 
	 * \brief The one and only instance of CvCapture is accessed using 
	 *        \e CvCapture::Instance()
	 */
    static CvCapture &Instance();
    /** 
	 * \brief Set the videocallback function that will be called for
	 *        every frame.
	 */
    void SetVideoCallback(void (*_videocallback)(IplImage *image));
    /** 
	 * \brief Sets the keyboard callback function that will be called 
	 *        when keyboard is pressed. 
	 *
	 * The callback should return 0 if it doesn't want the default keyboard 
	 * actions to be made. By default keys '0'-'9' executes 
	 * \e ToggleImageVisible for first 10 IplImage's, while any other key 
	 * will Exit the program.
	 */
    void SetKeyCallback(int (*_keycallback)(int key));
    /** 
	 * \brief Start video input from given capture device
	 * \param cap The capture device. If NULL a default capture device is created.
	 */
    bool StartVideo(Capture *_cap, const char *_wintitle = 0 /*"Capture"*/);
    /** 
	 * \brief Stop video
	 */
    void StopVideo()
    {
        running = false;
    }
    /** 
	 * \brief Sets an existing IplImage to be stored with the given title
	 * \param title Title for the image
	 * \param ipl The IplImage to be stored
	 * \param release_at_exit Boolean indicating should \e CvCapture automatically release the image at exit
	 */
    size_t SetImage(const char *title, IplImage *ipl, bool release_at_exit = false);
    /** 
	 * \brief Creates an image with given size, depth and channels and stores 
	 *        it with a given 'title' (see \e CvCapture::SetImage)
	 */
    IplImage *CreateImage(const char *title, CvSize size, int depth, int channels);
    /** 
	 * \brief Creates an image based on the given prototype and stores
	 *        it with a given 'title' (see \e CvCapture::SetImage)
	 */
    IplImage *CreateImageWithProto(const char *title, IplImage *proto, int depth = 0, int channels = 0);
    /** 
	 * \brief Get a pointer for the stored image based on index number
	 */
    IplImage *GetImage(size_t index);
    /** 
	 * \brief Get an index number of the stored image based on title
	 */
    size_t GetImageIndex(const char *title);
    /** 
	 * \brief Get a pointer for the stored image based on title
	 */
    IplImage *GetImage(const char *title);
    /** 
	 * \brief Toggle the visibility of the stored image
	 */
    bool ToggleImageVisible(size_t index, int flags = 1);
};

#endif
