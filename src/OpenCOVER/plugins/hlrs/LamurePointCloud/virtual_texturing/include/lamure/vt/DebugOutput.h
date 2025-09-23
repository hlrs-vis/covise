// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef LAMURE_DEBUGOUTPUT_H
#define LAMURE_DEBUGOUTPUT_H

class DebugOutput
{
public:
    static const int FPS_S = 60;
    static const int DISP_S = 60;
    static const int APPLY_S = 60;

    Debug() : _fps(FPS_S, 0.0f), _times_cut_dispatch(DISP_S, 0.0f), _times_apply(APPLY_S, 0.0f) {}
    ~Debug() = default;

    std::deque<float> &get_fps();
    std::deque<float> &get_cut_dispatch_times();
    std::deque<float> &get_apply_times();

private:
    std::deque<float> _fps;
    std::deque<float> _times_cut_dispatch;
    std::deque<float> _times_apply;
};

#endif //LAMURE_DEBUGOUTPUT_H
