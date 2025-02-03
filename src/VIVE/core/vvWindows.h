/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <util/coExport.h>

#include <vector>

namespace vive
{

    namespace ui {
        class Button;
    }

    class VVCORE_EXPORT vvWindows
    {

    private:
        static vvWindows* s_instance;

        int* origVSize, * origHSize;
        std::vector<int> oldWidth, oldHeight; // detect resized windows
        std::vector<int> origWidth, origHeight; // detect resized windows
        std::vector<float> aspectRatio;

        bool createWin(int i);
        bool destroyWin(int i);

        bool _firstTimeEmbedded;
        bool m_fullscreen = false;
        ui::Button* m_fullScreenButton = nullptr;

    public:
        vvWindows();

        ~vvWindows();

        bool config();
        void destroy();
        bool unconfig();

        void makeFullScreen(bool state);
        bool isFullScreen() const;

        void update();
        void updateContents();

        void setOrigVSize(int win, int size)
        {
            origVSize[win] = size;
        };
        void setOrigHSize(int win, int size)
        {
            origHSize[win] = size;
        };

        static vvWindows* instance();
    };
}
