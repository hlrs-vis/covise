// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef REN_CONFIG_H_
#define REN_CONFIG_H_

namespace lamure {
namespace ren {


//------------------------------
//for management:
//------------------------------


//#define LAMURE_ENABLE_INFO

//#define LAMURE_RENDERING_USE_SPLIT_SCREEN

//#define LAMURE_RENDERING_ENABLE_MULTI_VIEW_TEST
#define LAMURE_RENDERING_ENABLE_LAZY_MODELS_TEST

//#define LAMURE_CUT_UPDATE_ENABLE_MEASURE_SYSTEM_PERFORMANCE

#define LAMURE_DEFAULT_COLOR_R 1.0f
#define LAMURE_DEFAULT_COLOR_G 1.0f
#define LAMURE_DEFAULT_COLOR_B 1.0f

//------------------------------
//for cut_update_pool:
//------------------------------

#define LAMURE_CUT_UPDATE_ENABLE_MODEL_TIMEOUT
#define LAMURE_CUT_UPDATE_MAX_MODEL_TIMEOUT 18

//#define LAMURE_CUT_UPDATE_ENABLE_CUT_UPDATE_EXPERIMENTAL_MODE

//#define LAMURE_CUT_UPDATE_NUM_CUT_UPDATE_THREADS 4
#define LAMURE_CUT_UPDATE_NUM_CUT_UPDATE_THREADS 8

//#define LAMURE_CUT_UPDATE_ENABLE_SHOW_OOC_CACHE_USAGE
//#define LAMURE_CUT_UPDATE_ENABLE_SHOW_GPU_CACHE_USAGE

//allow multiple cut updates per frame
//#define LAMURE_CUT_UPDATE_ENABLE_REPEAT_MODE
#define LAMURE_CUT_UPDATE_MAX_NUM_UPDATES_PER_FRAME 3

#define LAMURE_CUT_UPDATE_ENABLE_SPLIT_AGAIN_MODE

#define LAMURE_CUT_UPDATE_MUST_COLLAPSE_OUTSIDE_FRUSTUM

#define LAMURE_DATABASE_SAFE_MODE

#define LAMURE_DEFAULT_IMPORTANCE 1.0f
#define LAMURE_MIN_IMPORTANCE 0.1f
#define LAMURE_MAX_IMPORTANCE 1.0f

#define LAMURE_DEFAULT_THRESHOLD 2.5f
#define LAMURE_MIN_THRESHOLD 1.0f
#define LAMURE_MAX_THRESHOLD 10.f

//#define LAMURE_CUT_UPDATE_ENABLE_PREFETCHING
#define LAMURE_CUT_UPDATE_PREFETCH_FACTOR 5.f
#define LAMURE_CUT_UPDATE_PREFETCH_BUDGET 1024

#define LAMURE_MIN_UPLOAD_BUDGET 16
#define LAMURE_MIN_VIDEO_MEMORY_BUDGET 128
#define LAMURE_MIN_MAIN_MEMORY_BUDGET 512
#define LAMURE_DEFAULT_UPLOAD_BUDGET 64
#define LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET 1024
#define LAMURE_DEFAULT_MAIN_MEMORY_BUDGET 4096
#define LAMURE_DEFAULT_SIZE_OF_PROVENANCE 0

//#define LAMURE_MIN_UPLOAD_BUDGET 16
//#define LAMURE_MIN_VIDEO_MEMORY_BUDGET 128
//#define LAMURE_MIN_MAIN_MEMORY_BUDGET 512
//#define LAMURE_DEFAULT_UPLOAD_BUDGET 64
//#define LAMURE_DEFAULT_VIDEO_MEMORY_BUDGET 1024
//#define LAMURE_DEFAULT_MAIN_MEMORY_BUDGET 4096
//#define LAMURE_DEFAULT_SIZE_OF_PROVENANCE 0

//------------------------------
//for ooc_cache:
//------------------------------

#define LAMURE_CUT_UPDATE_NUM_LOADING_THREADS 8
//#define LAMURE_CUT_UPDATE_NUM_LOADING_THREADS 24

#define LAMURE_CUT_UPDATE_ENABLE_CACHE_MAINTENANCE
#define LAMURE_CUT_UPDATE_CACHE_MAINTENANCE_COUNTER 60

//------------------------------
//for ooc_pool:
//------------------------------

//#define LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE cache_queue::update_mode::UPDATE_ALWAYS
#define LAMURE_CUT_UPDATE_LOADING_QUEUE_MODE cache_queue::update_mode::UPDATE_INCREMENT_ONLY

//------------------------------
//for bvh_stream: 
//------------------------------

//------------------------------
//for ray:
//------------------------------
#define LAMURE_WYSIWYG_SPLAT_SCALE 1.3f

#ifdef LAMURE_CUT_UPDATE_ENABLE_CUT_UPDATE_EXPERIMENTAL_MODE
#undef LAMURE_CUT_UPDATE_ENABLE_SPLIT_AGAIN_MODE
#endif
  


} } // namespace lamure


#endif // REN_CACHE_H_
