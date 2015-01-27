/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* this ALWAYS GENERATED file contains the definitions for the interfaces */

/* File created by MIDL compiler version 6.00.0366 */
/* Compiler settings for optitrack.idl:
    Oicf, W1, Zp8, env=Win32 (32b run)
    protocol : dce , ms_ext, c_ext, robust
    error checks: allocation ref bounds_check enum stub_data , no_format_optimization
    VC __declspec() decoration level: 
         __declspec(uuid()), __declspec(selectany), __declspec(novtable)
         DECLSPEC_UUID(), MIDL_INTERFACE()
*/
//@@MIDL_FILE_HEADING(  )

#pragma warning(disable : 4049) /* more than 64k source lines */

/* verify that the <rpcndr.h> version is high enough to compile this file*/
#ifndef __REQUIRED_RPCNDR_H_VERSION__
#define __REQUIRED_RPCNDR_H_VERSION__ 475
#endif

#include "rpc.h"
#include "rpcndr.h"

#ifndef __RPCNDR_H_VERSION__
#error this stub requires an updated version of <rpcndr.h>
#endif // __RPCNDR_H_VERSION__

#ifndef COM_NO_WINDOWS_H
#include "windows.h"
#include "ole2.h"
#endif /*COM_NO_WINDOWS_H*/

#ifndef __optitrack_h__
#define __optitrack_h__

#if defined(_MSC_VER) && (_MSC_VER >= 1020)
#pragma once
#endif

/* Forward Declarations */

#ifndef __INPCameraCollection_FWD_DEFINED__
#define __INPCameraCollection_FWD_DEFINED__
typedef interface INPCameraCollection INPCameraCollection;
#endif /* __INPCameraCollection_FWD_DEFINED__ */

#ifndef __INPCamera_FWD_DEFINED__
#define __INPCamera_FWD_DEFINED__
typedef interface INPCamera INPCamera;
#endif /* __INPCamera_FWD_DEFINED__ */

#ifndef __INPCameraFrame_FWD_DEFINED__
#define __INPCameraFrame_FWD_DEFINED__
typedef interface INPCameraFrame INPCameraFrame;
#endif /* __INPCameraFrame_FWD_DEFINED__ */

#ifndef __INPObject_FWD_DEFINED__
#define __INPObject_FWD_DEFINED__
typedef interface INPObject INPObject;
#endif /* __INPObject_FWD_DEFINED__ */

#ifndef __INPSmoothing_FWD_DEFINED__
#define __INPSmoothing_FWD_DEFINED__
typedef interface INPSmoothing INPSmoothing;
#endif /* __INPSmoothing_FWD_DEFINED__ */

#ifndef __INPVector_FWD_DEFINED__
#define __INPVector_FWD_DEFINED__
typedef interface INPVector INPVector;
#endif /* __INPVector_FWD_DEFINED__ */

#ifndef __INPPoint_FWD_DEFINED__
#define __INPPoint_FWD_DEFINED__
typedef interface INPPoint INPPoint;
#endif /* __INPPoint_FWD_DEFINED__ */

#ifndef __INPVector2_FWD_DEFINED__
#define __INPVector2_FWD_DEFINED__
typedef interface INPVector2 INPVector2;
#endif /* __INPVector2_FWD_DEFINED__ */

#ifndef __INPVector3_FWD_DEFINED__
#define __INPVector3_FWD_DEFINED__
typedef interface INPVector3 INPVector3;
#endif /* __INPVector3_FWD_DEFINED__ */

#ifndef __INPAvi_FWD_DEFINED__
#define __INPAvi_FWD_DEFINED__
typedef interface INPAvi INPAvi;
#endif /* __INPAvi_FWD_DEFINED__ */

#ifndef ___INPCameraCollectionEvents_FWD_DEFINED__
#define ___INPCameraCollectionEvents_FWD_DEFINED__
typedef interface _INPCameraCollectionEvents _INPCameraCollectionEvents;
#endif /* ___INPCameraCollectionEvents_FWD_DEFINED__ */

#ifndef ___INPCameraEvents_FWD_DEFINED__
#define ___INPCameraEvents_FWD_DEFINED__
typedef interface _INPCameraEvents _INPCameraEvents;
#endif /* ___INPCameraEvents_FWD_DEFINED__ */

#ifndef __NPCameraCollection_FWD_DEFINED__
#define __NPCameraCollection_FWD_DEFINED__

#ifdef __cplusplus
typedef class NPCameraCollection NPCameraCollection;
#else
typedef struct NPCameraCollection NPCameraCollection;
#endif /* __cplusplus */

#endif /* __NPCameraCollection_FWD_DEFINED__ */

#ifndef __NPCamera_FWD_DEFINED__
#define __NPCamera_FWD_DEFINED__

#ifdef __cplusplus
typedef class NPCamera NPCamera;
#else
typedef struct NPCamera NPCamera;
#endif /* __cplusplus */

#endif /* __NPCamera_FWD_DEFINED__ */

#ifndef __NPCameraFrame_FWD_DEFINED__
#define __NPCameraFrame_FWD_DEFINED__

#ifdef __cplusplus
typedef class NPCameraFrame NPCameraFrame;
#else
typedef struct NPCameraFrame NPCameraFrame;
#endif /* __cplusplus */

#endif /* __NPCameraFrame_FWD_DEFINED__ */

#ifndef __NPObject_FWD_DEFINED__
#define __NPObject_FWD_DEFINED__

#ifdef __cplusplus
typedef class NPObject NPObject;
#else
typedef struct NPObject NPObject;
#endif /* __cplusplus */

#endif /* __NPObject_FWD_DEFINED__ */

#ifndef __NPSmoothing_FWD_DEFINED__
#define __NPSmoothing_FWD_DEFINED__

#ifdef __cplusplus
typedef class NPSmoothing NPSmoothing;
#else
typedef struct NPSmoothing NPSmoothing;
#endif /* __cplusplus */

#endif /* __NPSmoothing_FWD_DEFINED__ */

#ifndef __NPVector_FWD_DEFINED__
#define __NPVector_FWD_DEFINED__

#ifdef __cplusplus
typedef class NPVector NPVector;
#else
typedef struct NPVector NPVector;
#endif /* __cplusplus */

#endif /* __NPVector_FWD_DEFINED__ */

#ifndef __NPPoint_FWD_DEFINED__
#define __NPPoint_FWD_DEFINED__

#ifdef __cplusplus
typedef class NPPoint NPPoint;
#else
typedef struct NPPoint NPPoint;
#endif /* __cplusplus */

#endif /* __NPPoint_FWD_DEFINED__ */

#ifndef __NPAvi_FWD_DEFINED__
#define __NPAvi_FWD_DEFINED__

#ifdef __cplusplus
typedef class NPAvi NPAvi;
#else
typedef struct NPAvi NPAvi;
#endif /* __cplusplus */

#endif /* __NPAvi_FWD_DEFINED__ */

/* header files for imported files */
#include "oaidl.h"
#include "ocidl.h"

#ifdef __cplusplus
extern "C" {
#endif

void *__RPC_USER MIDL_user_allocate(size_t);
void __RPC_USER MIDL_user_free(void *);

/* interface __MIDL_itf_optitrack_0000 */
/* [local] */

#define B2VB(b) ((VARIANT_BOOL)(b ? VARIANT_TRUE : VARIANT_FALSE))
#define VB2B(b) ((BOOL)b == VARIANT_TRUE ? TRUE : FALSE)
#define NP_THRESHOLD_MIN 1
#define NP_THRESHOLD_MAX 253
#define NP_DRAW_SCALE_MIN 0.1
#define NP_DRAW_SCALE_MAX 15.0
#define NP_SMOOTHING_MIN 10
#define NP_SMOOTHING_MAX 120
#define NP_E_DEVICE_NOT_SUPPORTED MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0xA000)
#define NP_E_DEVICE_DISCONNECTED MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0xA001)
#define NP_E_NEVER MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0xA002)
#define NP_E_WAIT MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0xA003)
#define NP_E_NOT_STARTED MAKE_HRESULT(SEVERITY_ERROR, FACILITY_ITF, 0xA004)
typedef /* [public] */
    enum __MIDL___MIDL_itf_optitrack_0000_0001
{
    NP_OPTION_STATUS_GREEN_ON_TRACKING = 0,
    NP_OPTION_TRACKED_OBJECT_COLOR = NP_OPTION_STATUS_GREEN_ON_TRACKING + 1,
    NP_OPTION_UNTRACKED_OBJECTS_COLOR = NP_OPTION_TRACKED_OBJECT_COLOR + 1,
    NP_OPTION_OBJECT_COLOR_OPTION = NP_OPTION_UNTRACKED_OBJECTS_COLOR + 1,
    NP_OPTION_DRAW_SCALE = NP_OPTION_OBJECT_COLOR_OPTION + 1,
    NP_OPTION_THRESHOLD = NP_OPTION_DRAW_SCALE + 1,
    NP_OPTION_OBJECT_MASS_WEIGHT = NP_OPTION_THRESHOLD + 1,
    NP_OPTION_OBJECT_RATIO_WEIGHT = NP_OPTION_OBJECT_MASS_WEIGHT + 1,
    NP_OPTION_PROXIMITY_WEIGHT = NP_OPTION_OBJECT_RATIO_WEIGHT + 1,
    NP_OPTION_STATIC_COUNT_WEIGHT = NP_OPTION_PROXIMITY_WEIGHT + 1,
    NP_OPTION_SCREEN_CENTER_WEIGHT = NP_OPTION_STATIC_COUNT_WEIGHT + 1,
    NP_OPTION_LAST_OBJECT_TRACKED_WEIGHT = NP_OPTION_SCREEN_CENTER_WEIGHT + 1,
    NP_OPTION_OBJECT_MASS_MIN = NP_OPTION_LAST_OBJECT_TRACKED_WEIGHT + 1,
    NP_OPTION_OBJECT_MASS_MAX = NP_OPTION_OBJECT_MASS_MIN + 1,
    NP_OPTION_OBJECT_MASS_IDEAL = NP_OPTION_OBJECT_MASS_MAX + 1,
    NP_OPTION_OBJECT_MASS_OUT_OF_RANGE = NP_OPTION_OBJECT_MASS_IDEAL + 1,
    NP_OPTION_OBJECT_RATIO_MIN = NP_OPTION_OBJECT_MASS_OUT_OF_RANGE + 1,
    NP_OPTION_OBJECT_RATIO_MAX = NP_OPTION_OBJECT_RATIO_MIN + 1,
    NP_OPTION_OBJECT_RATIO_IDEAL = NP_OPTION_OBJECT_RATIO_MAX + 1,
    NP_OPTION_OBJECT_RATIO_OUT_OF_RANGE = NP_OPTION_OBJECT_RATIO_IDEAL + 1,
    NP_OPTION_PROXIMITY_MIN = NP_OPTION_OBJECT_RATIO_OUT_OF_RANGE + 1,
    NP_OPTION_PROXIMITY_MAX = NP_OPTION_PROXIMITY_MIN + 1,
    NP_OPTION_PROXIMITY_IDEAL = NP_OPTION_PROXIMITY_MAX + 1,
    NP_OPTION_PROXIMITY_OUT_OF_RANGE = NP_OPTION_PROXIMITY_IDEAL + 1,
    NP_OPTION_STATIC_COUNT_MIN = NP_OPTION_PROXIMITY_OUT_OF_RANGE + 1,
    NP_OPTION_STATIC_COUNT_MAX = NP_OPTION_STATIC_COUNT_MIN + 1,
    NP_OPTION_STATIC_COUNT_IDEAL = NP_OPTION_STATIC_COUNT_MAX + 1,
    NP_OPTION_STATIC_COUNT_OUT_OF_RANGE = NP_OPTION_STATIC_COUNT_IDEAL + 1,
    NP_OPTION_SCREEN_CENTER_MIN = NP_OPTION_STATIC_COUNT_OUT_OF_RANGE + 1,
    NP_OPTION_SCREEN_CENTER_MAX = NP_OPTION_SCREEN_CENTER_MIN + 1,
    NP_OPTION_SCREEN_CENTER_IDEAL = NP_OPTION_SCREEN_CENTER_MAX + 1,
    NP_OPTION_SCREEN_CENTER_OUT_OF_RANGE = NP_OPTION_SCREEN_CENTER_IDEAL + 1,
    NP_OPTION_LAST_OBJECT_MIN = NP_OPTION_SCREEN_CENTER_OUT_OF_RANGE + 1,
    NP_OPTION_LAST_OBJECT_MAX = NP_OPTION_LAST_OBJECT_MIN + 1,
    NP_OPTION_LAST_OBJECT_IDEAL = NP_OPTION_LAST_OBJECT_MAX + 1,
    NP_OPTION_LAST_OBJECT_OUT_OF_RANGE = NP_OPTION_LAST_OBJECT_IDEAL + 1,
    NP_OPTION_STATUS_LED_ON_START = NP_OPTION_LAST_OBJECT_OUT_OF_RANGE + 1,
    NP_OPTION_ILLUMINATION_LEDS_ON_START = NP_OPTION_STATUS_LED_ON_START + 1,
    NP_OPTION_CAMERA_ROTATION = NP_OPTION_ILLUMINATION_LEDS_ON_START + 1,
    NP_OPTION_MIRROR_X = NP_OPTION_CAMERA_ROTATION + 1,
    NP_OPTION_MIRROR_Y = NP_OPTION_MIRROR_X + 1,
    NP_OPTION_SEND_EMPTY_FRAMES = NP_OPTION_MIRROR_Y + 1,
    NP_OPTION_CAMERA_ID = NP_OPTION_SEND_EMPTY_FRAMES + 1,
    NP_OPTION_CAMERA_ID_DEFAULT = NP_OPTION_CAMERA_ID + 1,
    NP_OPTION_FRAME_RATE = NP_OPTION_CAMERA_ID_DEFAULT + 1,
    NP_OPTION_FRAME_RATE_DEFAULT = NP_OPTION_FRAME_RATE + 1,
    NP_OPTION_EXPOSURE = NP_OPTION_FRAME_RATE_DEFAULT + 1,
    NP_OPTION_EXPOSURE_DEFAULT = NP_OPTION_EXPOSURE + 1,
    NP_OPTION_VIDEO_TYPE = NP_OPTION_EXPOSURE_DEFAULT + 1,
    NP_OPTION_VIDEO_TYPE_DEFAULT = NP_OPTION_VIDEO_TYPE + 1,
    NP_OPTION_INTENSITY = NP_OPTION_VIDEO_TYPE_DEFAULT + 1,
    NP_OPTION_INTENSITY_DEFAULT = NP_OPTION_INTENSITY + 1,
    NP_OPTION_FRAME_DECIMATION = NP_OPTION_INTENSITY_DEFAULT + 1,
    NP_OPTION_FRAME_DECIMATION_DEFAULT = NP_OPTION_FRAME_DECIMATION + 1,
    NP_OPTION_MINIMUM_SEGMENT_LENGTH = NP_OPTION_FRAME_DECIMATION_DEFAULT + 1,
    NP_OPTION_MINIMUM_SEGMENT_LENGTH_DEFAULT = NP_OPTION_MINIMUM_SEGMENT_LENGTH + 1,
    NP_OPTION_MAXIMUM_SEGMENT_LENGTH = NP_OPTION_MINIMUM_SEGMENT_LENGTH_DEFAULT + 1,
    NP_OPTION_MAXIMUM_SEGMENT_LENGTH_DEFAULT = NP_OPTION_MAXIMUM_SEGMENT_LENGTH + 1,
    NP_OPTION_WINDOW_EXTENTS_X = NP_OPTION_MAXIMUM_SEGMENT_LENGTH_DEFAULT + 1,
    NP_OPTION_WINDOW_EXTENTS_X_DEFAULT = NP_OPTION_WINDOW_EXTENTS_X + 1,
    NP_OPTION_WINDOW_EXTENTS_X_END = NP_OPTION_WINDOW_EXTENTS_X_DEFAULT + 1,
    NP_OPTION_WINDOW_EXTENTS_X_END_DEFAULT = NP_OPTION_WINDOW_EXTENTS_X_END + 1,
    NP_OPTION_WINDOW_EXTENTS_Y = NP_OPTION_WINDOW_EXTENTS_X_END_DEFAULT + 1,
    NP_OPTION_WINDOW_EXTENTS_Y_DEFAULT = NP_OPTION_WINDOW_EXTENTS_Y + 1,
    NP_OPTION_WINDOW_EXTENTS_Y_END = NP_OPTION_WINDOW_EXTENTS_Y_DEFAULT + 1,
    NP_OPTION_WINDOW_EXTENTS_Y_END_DEFAULT = NP_OPTION_WINDOW_EXTENTS_Y_END + 1,
    NP_OPTION_RESET_FRAME_COUNT = NP_OPTION_WINDOW_EXTENTS_Y_END_DEFAULT + 1,
    NP_OPTION_USER_HWND = NP_OPTION_RESET_FRAME_COUNT + 1,
    NP_OPTION_MULTICAM = NP_OPTION_USER_HWND + 1,
    NP_OPTION_MULTICAM_MASTER = NP_OPTION_MULTICAM + 1,
    NP_OPTION_MULTICAM_GROUP_NOTIFY = NP_OPTION_MULTICAM_MASTER + 1,
    NP_OPTION_NUMERIC_DISPLAY_ON = NP_OPTION_MULTICAM_GROUP_NOTIFY + 1,
    NP_OPTION_NUMERIC_DISPLAY_OFF = NP_OPTION_NUMERIC_DISPLAY_ON + 1,
    NP_OPTION_SEND_FRAME_MASK = NP_OPTION_NUMERIC_DISPLAY_OFF + 1,
    NP_OPTION_TEXT_OVERLAY_OPTION = NP_OPTION_SEND_FRAME_MASK + 1,
    NP_OPTION_USER_DEF1 = NP_OPTION_TEXT_OVERLAY_OPTION + 1,
    NP_OPTION_SCORING_ENABLED = NP_OPTION_USER_DEF1 + 1,
    NP_OPTION_GRAYSCALE_DECIMATION = NP_OPTION_SCORING_ENABLED + 1,
    NP_OPTION_OBJECT_CAP = NP_OPTION_GRAYSCALE_DECIMATION + 1,
    NP_OPTION_HARDWARE_REVISION = NP_OPTION_OBJECT_CAP + 1,
    NP_OPTION_SENSORCLOCK_FORCEENABLE = NP_OPTION_HARDWARE_REVISION + 1,
    NP_OPTION_SHUTTER_DELAY = NP_OPTION_SENSORCLOCK_FORCEENABLE + 1,
    NP_OPTION_SYNC_MODE = NP_OPTION_SHUTTER_DELAY + 1,
    NP_OPTION_TRIGGER_SNAPSHOT = NP_OPTION_SYNC_MODE + 1,
    NP_OPTION_RESET_SNAPTSHOT_ID = NP_OPTION_TRIGGER_SNAPSHOT + 1,
    NP_OPTION_ALLOW_CLOCK_PAUSE_ENABLE = NP_OPTION_RESET_SNAPTSHOT_ID + 1,
    NP_OPTION_SET_IR_FILTER = NP_OPTION_ALLOW_CLOCK_PAUSE_ENABLE + 1
} NP_OPTION;

typedef /* [public] */
    enum __MIDL___MIDL_itf_optitrack_0000_0002
{
    NP_SYNC_NORMAL = 0,
    NP_SYNC_FORCE_SLAVE = 0x1,
    NP_SYNC_SNAPSHOT_PSEUDO_MASTER = 0x2,
    NP_SYNC_FORCE_SLAVE_LAST = 0x3
} NP_SYNC_MODE_LIST;

typedef /* [public] */
    enum __MIDL___MIDL_itf_optitrack_0000_0003
{
    NP_FRAME_SENDEMPTY = 0x1,
    NP_FRAME_SENDINVALID = 0x2,
    NP_FRAME_GRAYSCALEOBJECTS = 0x4,
    NP_FRAME_GRAYSCALEPRECISION = 0x8
} NP_FRAME_SEND_MASK;

typedef /* [public] */
    enum __MIDL___MIDL_itf_optitrack_0000_0004
{
    NP_HW_MODEL_OLDTRACKIR = 0x100800a8,
    NP_HW_MODEL_SMARTNAV = NP_HW_MODEL_OLDTRACKIR + 1,
    NP_HW_MODEL_TRACKIR = NP_HW_MODEL_SMARTNAV + 1,
    NP_HW_MODEL_OPTITRACK = NP_HW_MODEL_TRACKIR + 1,
    NP_HW_MODEL_UNKNOWN = NP_HW_MODEL_OPTITRACK + 1
} NP_HW_MODEL;

typedef /* [public] */
    enum __MIDL___MIDL_itf_optitrack_0000_0005
{
    NP_HW_REVISION_OLDTRACKIR_LEGACY = 0x200800b8,
    NP_HW_REVISION_OLDTRACKIR_BASIC = NP_HW_REVISION_OLDTRACKIR_LEGACY + 1,
    NP_HW_REVISION_OLDTRACKIR_EG = NP_HW_REVISION_OLDTRACKIR_BASIC + 1,
    NP_HW_REVISION_OLDTRACKIR_AT = NP_HW_REVISION_OLDTRACKIR_EG + 1,
    NP_HW_REVISION_OLDTRACKIR_GX = NP_HW_REVISION_OLDTRACKIR_AT + 1,
    NP_HW_REVISION_OLDTRACKIR_MAC = NP_HW_REVISION_OLDTRACKIR_GX + 1,
    NP_HW_REVISION_SMARTNAV_BASIC = NP_HW_REVISION_OLDTRACKIR_MAC + 1,
    NP_HW_REVISION_SMARTNAV_EG = NP_HW_REVISION_SMARTNAV_BASIC + 1,
    NP_HW_REVISION_SMARTNAV_AT = NP_HW_REVISION_SMARTNAV_EG + 1,
    NP_HW_REVISION_SMARTNAV_MAC_BASIC = NP_HW_REVISION_SMARTNAV_AT + 1,
    NP_HW_REVISION_SMARTNAV_MAC_AT = NP_HW_REVISION_SMARTNAV_MAC_BASIC + 1,
    NP_HW_REVISION_TRACKIR_BASIC = NP_HW_REVISION_SMARTNAV_MAC_AT + 1,
    NP_HW_REVISION_TRACKIR_PRO = NP_HW_REVISION_TRACKIR_BASIC + 1,
    NP_HW_REVISION_OPTITRACK_BASIC = NP_HW_REVISION_TRACKIR_PRO + 1,
    NP_HW_REVISION_OPTITRACK_FLEX = NP_HW_REVISION_OPTITRACK_BASIC + 1,
    NP_HW_REVISION_OPTITRACK_SLIM = NP_HW_REVISION_OPTITRACK_FLEX + 1,
    NP_HW_REVISION_OPTITRACK_FLEX_FILTERSWITCH = NP_HW_REVISION_OPTITRACK_SLIM + 1,
    NP_HW_REVISION_OPTITRACK_SLIM_FILTERSWITCH = NP_HW_REVISION_OPTITRACK_FLEX_FILTERSWITCH + 1,
    NP_HW_REVISION_UNKNOWN = NP_HW_REVISION_OPTITRACK_SLIM_FILTERSWITCH + 1
} NP_HW_REVISION;

typedef /* [public] */
    enum __MIDL___MIDL_itf_optitrack_0000_0006
{
    NP_OBJECT_COLOR_OPTION_ALL_SAME = 0,
    NP_OBJECT_COLOR_OPTION_TRACKED_ONLY = NP_OBJECT_COLOR_OPTION_ALL_SAME + 1,
    NP_OBJECT_COLOR_OPTION_SHADE_BY_RANK = NP_OBJECT_COLOR_OPTION_TRACKED_ONLY + 1,
    NP_OBJECT_COLOR_OPTION_VECTOR = NP_OBJECT_COLOR_OPTION_SHADE_BY_RANK + 1,
    NP_OBJECT_COLOR_OPTION_VECTOR_SHADE_BY_RANK = NP_OBJECT_COLOR_OPTION_VECTOR + 1
} NP_OBJECT_COLOR_OPTION;

typedef /* [public] */
    enum __MIDL___MIDL_itf_optitrack_0000_0007
{
    NP_TEXT_OVERLAY_HEADER = 0x1,
    NP_TEXT_OVERLAY_OBJECT = 0x2,
    NP_TEXT_OVERLAY_OBJECT_HIGHLIGHT = 0x4,
    NP_TEXT_OVERLAY_PRECISION_GRAYSCALE = 0x8
} NP_TEXT_OVERLAY_OPTION;

typedef /* [public] */
    enum __MIDL___MIDL_itf_optitrack_0000_0008
{
    NP_SWITCH_STATE_BOTH_DOWN = 0,
    NP_SWITCH_STATE_ONE_DOWN = 0x4,
    NP_SWITCH_STATE_TWO_DOWN = 0x8,
    NP_SWITCH_STATE_BOTH_UP = 0xc
} NP_SWITCH_STATE;

typedef /* [public] */
    enum __MIDL___MIDL_itf_optitrack_0000_0009
{
    NP_LED_ONE = 0,
    NP_LED_TWO = NP_LED_ONE + 1,
    NP_LED_THREE = NP_LED_TWO + 1,
    NP_LED_FOUR = NP_LED_THREE + 1
} NP_LED;

typedef /* [public] */
    enum __MIDL___MIDL_itf_optitrack_0000_0010
{
    NP_CAMERA_ROTATION_0 = 0,
    NP_CAMERA_ROTATION_90 = NP_CAMERA_ROTATION_0 + 1,
    NP_CAMERA_ROTATION_180 = NP_CAMERA_ROTATION_90 + 1,
    NP_CAMERA_ROTATION_270 = NP_CAMERA_ROTATION_180 + 1
} NP_CAMERA_ROTATION;

extern RPC_IF_HANDLE __MIDL_itf_optitrack_0000_v0_0_c_ifspec;
extern RPC_IF_HANDLE __MIDL_itf_optitrack_0000_v0_0_s_ifspec;

#ifndef __INPCameraCollection_INTERFACE_DEFINED__
#define __INPCameraCollection_INTERFACE_DEFINED__

/* interface INPCameraCollection */
/* [unique][helpstring][dual][uuid][object] */

EXTERN_C const IID IID_INPCameraCollection;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("28E501BB-FDD9-46CF-A112-741587110F0E")
INPCameraCollection:
public
IDispatch
{
public:
    virtual /* [restricted][helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get__NewEnum(
        /* [retval][out] */ LPUNKNOWN * ppunk) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Count(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Item(
        /* [in] */ LONG a_vlIndex,
        /* [retval][out] */ INPCamera * *ppCamera) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Enum(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Synchronize(void) = 0;
};

#else /* C style interface */

typedef struct INPCameraCollectionVtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        INPCameraCollection *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        INPCameraCollection *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        INPCameraCollection *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        INPCameraCollection *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        INPCameraCollection *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        INPCameraCollection *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        INPCameraCollection *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    /* [restricted][helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get__NewEnum)(
        INPCameraCollection *This,
        /* [retval][out] */ LPUNKNOWN *ppunk);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Count)(
        INPCameraCollection *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Item)(
        INPCameraCollection *This,
        /* [in] */ LONG a_vlIndex,
        /* [retval][out] */ INPCamera **ppCamera);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Enum)(
        INPCameraCollection *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Synchronize)(
        INPCameraCollection *This);

    END_INTERFACE
} INPCameraCollectionVtbl;

interface INPCameraCollection
{
    CONST_VTBL struct INPCameraCollectionVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define INPCameraCollection_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define INPCameraCollection_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define INPCameraCollection_Release(This) \
    (This)->lpVtbl->Release(This)

#define INPCameraCollection_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define INPCameraCollection_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define INPCameraCollection_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define INPCameraCollection_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#define INPCameraCollection_get__NewEnum(This, ppunk) \
    (This)->lpVtbl->get__NewEnum(This, ppunk)

#define INPCameraCollection_get_Count(This, pVal) \
    (This)->lpVtbl->get_Count(This, pVal)

#define INPCameraCollection_Item(This, a_vlIndex, ppCamera) \
    (This)->lpVtbl->Item(This, a_vlIndex, ppCamera)

#define INPCameraCollection_Enum(This) \
    (This)->lpVtbl->Enum(This)

#define INPCameraCollection_Synchronize(This) \
    (This)->lpVtbl->Synchronize(This)

#endif /* COBJMACROS */

#endif /* C style interface */

/* [restricted][helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraCollection_get__NewEnum_Proxy(
    INPCameraCollection *This,
    /* [retval][out] */ LPUNKNOWN *ppunk);

void __RPC_STUB INPCameraCollection_get__NewEnum_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraCollection_get_Count_Proxy(
    INPCameraCollection *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCameraCollection_get_Count_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCameraCollection_Item_Proxy(
    INPCameraCollection *This,
    /* [in] */ LONG a_vlIndex,
    /* [retval][out] */ INPCamera **ppCamera);

void __RPC_STUB INPCameraCollection_Item_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCameraCollection_Enum_Proxy(
    INPCameraCollection *This);

void __RPC_STUB INPCameraCollection_Enum_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCameraCollection_Synchronize_Proxy(
    INPCameraCollection *This);

void __RPC_STUB INPCameraCollection_Synchronize_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

#endif /* __INPCameraCollection_INTERFACE_DEFINED__ */

#ifndef __INPCamera_INTERFACE_DEFINED__
#define __INPCamera_INTERFACE_DEFINED__

/* interface INPCamera */
/* [unique][helpstring][dual][uuid][object] */

EXTERN_C const IID IID_INPCamera;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("ADE1E272-C86A-460D-B7B9-3051F310E4D0")
INPCamera:
public
IDispatch
{
public:
    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_SerialNumber(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Model(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Revision(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Width(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Height(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_FrameRate(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_BlockingMaskWidth(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_BlockingMaskHeight(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Start(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Stop(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Open(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Close(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE SetLED(
        /* [in] */ LONG lLED,
        /* [in] */ VARIANT_BOOL fOn) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE GetFrame(
        /* [in] */ LONG lTimeout,
        /* [retval][out] */ INPCameraFrame * *ppFrame) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE DrawFrame(
        /* [in] */ INPCameraFrame * pFrame,
        /* [in] */ LONG hwnd) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE ResetTrackedObject(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE GetOption(
        /* [in] */ LONG lOption,
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE SetOption(
        /* [in] */ LONG lOption,
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE GetFrameById(
        /* [in] */ LONG Id,
        /* [retval][out] */ INPCameraFrame * *ppFrame) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE GetFrameImage(
        /* [in] */ INPCameraFrame * pFrame,
        /* [in] */ INT PixelWidth,
        /* [in] */ INT PixelHeight,
        /* [in] */ INT ByteSpan,
        /* [in] */ INT BitsPerPixel,
        /* [in] */ BYTE * Buffer) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE SetVideo(
        /* [in] */ VARIANT_BOOL fOn) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE ClearBlockingMask(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE EnableBlockingMask(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE DisableBlockingMask(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE AddBlockingRectangle(
        /* [in] */ LONG X1,
        /* [in] */ LONG Y1,
        /* [in] */ LONG X2,
        /* [in] */ LONG Y2) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE RemoveBlockingRectangle(
        /* [in] */ LONG X1,
        /* [in] */ LONG Y1,
        /* [in] */ LONG X2,
        /* [in] */ LONG Y2) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE GetBlockingMask(
        /* [in] */ BYTE * Buffer,
        /* [in] */ LONG BufferSize) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE SetBlockingMask(
        /* [in] */ BYTE * Buffer,
        /* [in] */ LONG BufferSize) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE UpdateBlockingMask(void) = 0;
};

#else /* C style interface */

typedef struct INPCameraVtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        INPCamera *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        INPCamera *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        INPCamera *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        INPCamera *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        INPCamera *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        INPCamera *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        INPCamera *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_SerialNumber)(
        INPCamera *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Model)(
        INPCamera *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Revision)(
        INPCamera *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Width)(
        INPCamera *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Height)(
        INPCamera *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_FrameRate)(
        INPCamera *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_BlockingMaskWidth)(
        INPCamera *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_BlockingMaskHeight)(
        INPCamera *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Start)(
        INPCamera *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Stop)(
        INPCamera *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Open)(
        INPCamera *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Close)(
        INPCamera *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *SetLED)(
        INPCamera *This,
        /* [in] */ LONG lLED,
        /* [in] */ VARIANT_BOOL fOn);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *GetFrame)(
        INPCamera *This,
        /* [in] */ LONG lTimeout,
        /* [retval][out] */ INPCameraFrame **ppFrame);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *DrawFrame)(
        INPCamera *This,
        /* [in] */ INPCameraFrame *pFrame,
        /* [in] */ LONG hwnd);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *ResetTrackedObject)(
        INPCamera *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *GetOption)(
        INPCamera *This,
        /* [in] */ LONG lOption,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *SetOption)(
        INPCamera *This,
        /* [in] */ LONG lOption,
        /* [in] */ VARIANT Val);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *GetFrameById)(
        INPCamera *This,
        /* [in] */ LONG Id,
        /* [retval][out] */ INPCameraFrame **ppFrame);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *GetFrameImage)(
        INPCamera *This,
        /* [in] */ INPCameraFrame *pFrame,
        /* [in] */ INT PixelWidth,
        /* [in] */ INT PixelHeight,
        /* [in] */ INT ByteSpan,
        /* [in] */ INT BitsPerPixel,
        /* [in] */ BYTE *Buffer);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *SetVideo)(
        INPCamera *This,
        /* [in] */ VARIANT_BOOL fOn);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *ClearBlockingMask)(
        INPCamera *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *EnableBlockingMask)(
        INPCamera *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *DisableBlockingMask)(
        INPCamera *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *AddBlockingRectangle)(
        INPCamera *This,
        /* [in] */ LONG X1,
        /* [in] */ LONG Y1,
        /* [in] */ LONG X2,
        /* [in] */ LONG Y2);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *RemoveBlockingRectangle)(
        INPCamera *This,
        /* [in] */ LONG X1,
        /* [in] */ LONG Y1,
        /* [in] */ LONG X2,
        /* [in] */ LONG Y2);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *GetBlockingMask)(
        INPCamera *This,
        /* [in] */ BYTE *Buffer,
        /* [in] */ LONG BufferSize);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *SetBlockingMask)(
        INPCamera *This,
        /* [in] */ BYTE *Buffer,
        /* [in] */ LONG BufferSize);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *UpdateBlockingMask)(
        INPCamera *This);

    END_INTERFACE
} INPCameraVtbl;

interface INPCamera
{
    CONST_VTBL struct INPCameraVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define INPCamera_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define INPCamera_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define INPCamera_Release(This) \
    (This)->lpVtbl->Release(This)

#define INPCamera_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define INPCamera_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define INPCamera_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define INPCamera_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#define INPCamera_get_SerialNumber(This, pVal) \
    (This)->lpVtbl->get_SerialNumber(This, pVal)

#define INPCamera_get_Model(This, pVal) \
    (This)->lpVtbl->get_Model(This, pVal)

#define INPCamera_get_Revision(This, pVal) \
    (This)->lpVtbl->get_Revision(This, pVal)

#define INPCamera_get_Width(This, pVal) \
    (This)->lpVtbl->get_Width(This, pVal)

#define INPCamera_get_Height(This, pVal) \
    (This)->lpVtbl->get_Height(This, pVal)

#define INPCamera_get_FrameRate(This, pVal) \
    (This)->lpVtbl->get_FrameRate(This, pVal)

#define INPCamera_get_BlockingMaskWidth(This, pVal) \
    (This)->lpVtbl->get_BlockingMaskWidth(This, pVal)

#define INPCamera_get_BlockingMaskHeight(This, pVal) \
    (This)->lpVtbl->get_BlockingMaskHeight(This, pVal)

#define INPCamera_Start(This) \
    (This)->lpVtbl->Start(This)

#define INPCamera_Stop(This) \
    (This)->lpVtbl->Stop(This)

#define INPCamera_Open(This) \
    (This)->lpVtbl->Open(This)

#define INPCamera_Close(This) \
    (This)->lpVtbl->Close(This)

#define INPCamera_SetLED(This, lLED, fOn) \
    (This)->lpVtbl->SetLED(This, lLED, fOn)

#define INPCamera_GetFrame(This, lTimeout, ppFrame) \
    (This)->lpVtbl->GetFrame(This, lTimeout, ppFrame)

#define INPCamera_DrawFrame(This, pFrame, hwnd) \
    (This)->lpVtbl->DrawFrame(This, pFrame, hwnd)

#define INPCamera_ResetTrackedObject(This) \
    (This)->lpVtbl->ResetTrackedObject(This)

#define INPCamera_GetOption(This, lOption, pVal) \
    (This)->lpVtbl->GetOption(This, lOption, pVal)

#define INPCamera_SetOption(This, lOption, Val) \
    (This)->lpVtbl->SetOption(This, lOption, Val)

#define INPCamera_GetFrameById(This, Id, ppFrame) \
    (This)->lpVtbl->GetFrameById(This, Id, ppFrame)

#define INPCamera_GetFrameImage(This, pFrame, PixelWidth, PixelHeight, ByteSpan, BitsPerPixel, Buffer) \
    (This)->lpVtbl->GetFrameImage(This, pFrame, PixelWidth, PixelHeight, ByteSpan, BitsPerPixel, Buffer)

#define INPCamera_SetVideo(This, fOn) \
    (This)->lpVtbl->SetVideo(This, fOn)

#define INPCamera_ClearBlockingMask(This) \
    (This)->lpVtbl->ClearBlockingMask(This)

#define INPCamera_EnableBlockingMask(This) \
    (This)->lpVtbl->EnableBlockingMask(This)

#define INPCamera_DisableBlockingMask(This) \
    (This)->lpVtbl->DisableBlockingMask(This)

#define INPCamera_AddBlockingRectangle(This, X1, Y1, X2, Y2) \
    (This)->lpVtbl->AddBlockingRectangle(This, X1, Y1, X2, Y2)

#define INPCamera_RemoveBlockingRectangle(This, X1, Y1, X2, Y2) \
    (This)->lpVtbl->RemoveBlockingRectangle(This, X1, Y1, X2, Y2)

#define INPCamera_GetBlockingMask(This, Buffer, BufferSize) \
    (This)->lpVtbl->GetBlockingMask(This, Buffer, BufferSize)

#define INPCamera_SetBlockingMask(This, Buffer, BufferSize) \
    (This)->lpVtbl->SetBlockingMask(This, Buffer, BufferSize)

#define INPCamera_UpdateBlockingMask(This) \
    (This)->lpVtbl->UpdateBlockingMask(This)

#endif /* COBJMACROS */

#endif /* C style interface */

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCamera_get_SerialNumber_Proxy(
    INPCamera *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCamera_get_SerialNumber_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCamera_get_Model_Proxy(
    INPCamera *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCamera_get_Model_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCamera_get_Revision_Proxy(
    INPCamera *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCamera_get_Revision_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCamera_get_Width_Proxy(
    INPCamera *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCamera_get_Width_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCamera_get_Height_Proxy(
    INPCamera *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCamera_get_Height_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCamera_get_FrameRate_Proxy(
    INPCamera *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCamera_get_FrameRate_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCamera_get_BlockingMaskWidth_Proxy(
    INPCamera *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCamera_get_BlockingMaskWidth_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCamera_get_BlockingMaskHeight_Proxy(
    INPCamera *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCamera_get_BlockingMaskHeight_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_Start_Proxy(
    INPCamera *This);

void __RPC_STUB INPCamera_Start_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_Stop_Proxy(
    INPCamera *This);

void __RPC_STUB INPCamera_Stop_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_Open_Proxy(
    INPCamera *This);

void __RPC_STUB INPCamera_Open_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_Close_Proxy(
    INPCamera *This);

void __RPC_STUB INPCamera_Close_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_SetLED_Proxy(
    INPCamera *This,
    /* [in] */ LONG lLED,
    /* [in] */ VARIANT_BOOL fOn);

void __RPC_STUB INPCamera_SetLED_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_GetFrame_Proxy(
    INPCamera *This,
    /* [in] */ LONG lTimeout,
    /* [retval][out] */ INPCameraFrame **ppFrame);

void __RPC_STUB INPCamera_GetFrame_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_DrawFrame_Proxy(
    INPCamera *This,
    /* [in] */ INPCameraFrame *pFrame,
    /* [in] */ LONG hwnd);

void __RPC_STUB INPCamera_DrawFrame_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_ResetTrackedObject_Proxy(
    INPCamera *This);

void __RPC_STUB INPCamera_ResetTrackedObject_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_GetOption_Proxy(
    INPCamera *This,
    /* [in] */ LONG lOption,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPCamera_GetOption_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_SetOption_Proxy(
    INPCamera *This,
    /* [in] */ LONG lOption,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPCamera_SetOption_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_GetFrameById_Proxy(
    INPCamera *This,
    /* [in] */ LONG Id,
    /* [retval][out] */ INPCameraFrame **ppFrame);

void __RPC_STUB INPCamera_GetFrameById_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_GetFrameImage_Proxy(
    INPCamera *This,
    /* [in] */ INPCameraFrame *pFrame,
    /* [in] */ INT PixelWidth,
    /* [in] */ INT PixelHeight,
    /* [in] */ INT ByteSpan,
    /* [in] */ INT BitsPerPixel,
    /* [in] */ BYTE *Buffer);

void __RPC_STUB INPCamera_GetFrameImage_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_SetVideo_Proxy(
    INPCamera *This,
    /* [in] */ VARIANT_BOOL fOn);

void __RPC_STUB INPCamera_SetVideo_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_ClearBlockingMask_Proxy(
    INPCamera *This);

void __RPC_STUB INPCamera_ClearBlockingMask_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_EnableBlockingMask_Proxy(
    INPCamera *This);

void __RPC_STUB INPCamera_EnableBlockingMask_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_DisableBlockingMask_Proxy(
    INPCamera *This);

void __RPC_STUB INPCamera_DisableBlockingMask_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_AddBlockingRectangle_Proxy(
    INPCamera *This,
    /* [in] */ LONG X1,
    /* [in] */ LONG Y1,
    /* [in] */ LONG X2,
    /* [in] */ LONG Y2);

void __RPC_STUB INPCamera_AddBlockingRectangle_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_RemoveBlockingRectangle_Proxy(
    INPCamera *This,
    /* [in] */ LONG X1,
    /* [in] */ LONG Y1,
    /* [in] */ LONG X2,
    /* [in] */ LONG Y2);

void __RPC_STUB INPCamera_RemoveBlockingRectangle_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_GetBlockingMask_Proxy(
    INPCamera *This,
    /* [in] */ BYTE *Buffer,
    /* [in] */ LONG BufferSize);

void __RPC_STUB INPCamera_GetBlockingMask_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_SetBlockingMask_Proxy(
    INPCamera *This,
    /* [in] */ BYTE *Buffer,
    /* [in] */ LONG BufferSize);

void __RPC_STUB INPCamera_SetBlockingMask_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCamera_UpdateBlockingMask_Proxy(
    INPCamera *This);

void __RPC_STUB INPCamera_UpdateBlockingMask_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

#endif /* __INPCamera_INTERFACE_DEFINED__ */

#ifndef __INPCameraFrame_INTERFACE_DEFINED__
#define __INPCameraFrame_INTERFACE_DEFINED__

/* interface INPCameraFrame */
/* [unique][helpstring][dual][uuid][object] */

EXTERN_C const IID IID_INPCameraFrame;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("73CF9A64-837A-4F05-9BF6-8A253CE16E46")
INPCameraFrame:
public
IDispatch
{
public:
    virtual /* [restricted][helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get__NewEnum(
        /* [retval][out] */ LPUNKNOWN * ppunk) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Count(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Item(
        /* [in] */ LONG a_vlIndex,
        /* [retval][out] */ INPObject * *ppObject) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Id(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_SwitchState(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_IsEmpty(
        /* [retval][out] */ VARIANT_BOOL * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_IsCorrupt(
        /* [retval][out] */ VARIANT_BOOL * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_IsGreyscale(
        /* [retval][out] */ VARIANT_BOOL * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_TimeStamp(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_TimeStampFrequency(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE GetObjectData(
        /* [in] */ BYTE * Buffer,
        /* [in] */ INT BufferSize,
        /* [retval][out] */ LONG * pObjectCount) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Free(void) = 0;
};

#else /* C style interface */

typedef struct INPCameraFrameVtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        INPCameraFrame *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        INPCameraFrame *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        INPCameraFrame *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        INPCameraFrame *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        INPCameraFrame *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        INPCameraFrame *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        INPCameraFrame *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    /* [restricted][helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get__NewEnum)(
        INPCameraFrame *This,
        /* [retval][out] */ LPUNKNOWN *ppunk);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Count)(
        INPCameraFrame *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Item)(
        INPCameraFrame *This,
        /* [in] */ LONG a_vlIndex,
        /* [retval][out] */ INPObject **ppObject);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Id)(
        INPCameraFrame *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_SwitchState)(
        INPCameraFrame *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_IsEmpty)(
        INPCameraFrame *This,
        /* [retval][out] */ VARIANT_BOOL *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_IsCorrupt)(
        INPCameraFrame *This,
        /* [retval][out] */ VARIANT_BOOL *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_IsGreyscale)(
        INPCameraFrame *This,
        /* [retval][out] */ VARIANT_BOOL *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_TimeStamp)(
        INPCameraFrame *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_TimeStampFrequency)(
        INPCameraFrame *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *GetObjectData)(
        INPCameraFrame *This,
        /* [in] */ BYTE *Buffer,
        /* [in] */ INT BufferSize,
        /* [retval][out] */ LONG *pObjectCount);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Free)(
        INPCameraFrame *This);

    END_INTERFACE
} INPCameraFrameVtbl;

interface INPCameraFrame
{
    CONST_VTBL struct INPCameraFrameVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define INPCameraFrame_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define INPCameraFrame_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define INPCameraFrame_Release(This) \
    (This)->lpVtbl->Release(This)

#define INPCameraFrame_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define INPCameraFrame_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define INPCameraFrame_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define INPCameraFrame_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#define INPCameraFrame_get__NewEnum(This, ppunk) \
    (This)->lpVtbl->get__NewEnum(This, ppunk)

#define INPCameraFrame_get_Count(This, pVal) \
    (This)->lpVtbl->get_Count(This, pVal)

#define INPCameraFrame_Item(This, a_vlIndex, ppObject) \
    (This)->lpVtbl->Item(This, a_vlIndex, ppObject)

#define INPCameraFrame_get_Id(This, pVal) \
    (This)->lpVtbl->get_Id(This, pVal)

#define INPCameraFrame_get_SwitchState(This, pVal) \
    (This)->lpVtbl->get_SwitchState(This, pVal)

#define INPCameraFrame_get_IsEmpty(This, pVal) \
    (This)->lpVtbl->get_IsEmpty(This, pVal)

#define INPCameraFrame_get_IsCorrupt(This, pVal) \
    (This)->lpVtbl->get_IsCorrupt(This, pVal)

#define INPCameraFrame_get_IsGreyscale(This, pVal) \
    (This)->lpVtbl->get_IsGreyscale(This, pVal)

#define INPCameraFrame_get_TimeStamp(This, pVal) \
    (This)->lpVtbl->get_TimeStamp(This, pVal)

#define INPCameraFrame_get_TimeStampFrequency(This, pVal) \
    (This)->lpVtbl->get_TimeStampFrequency(This, pVal)

#define INPCameraFrame_GetObjectData(This, Buffer, BufferSize, pObjectCount) \
    (This)->lpVtbl->GetObjectData(This, Buffer, BufferSize, pObjectCount)

#define INPCameraFrame_Free(This) \
    (This)->lpVtbl->Free(This)

#endif /* COBJMACROS */

#endif /* C style interface */

/* [restricted][helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_get__NewEnum_Proxy(
    INPCameraFrame *This,
    /* [retval][out] */ LPUNKNOWN *ppunk);

void __RPC_STUB INPCameraFrame_get__NewEnum_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_get_Count_Proxy(
    INPCameraFrame *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCameraFrame_get_Count_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_Item_Proxy(
    INPCameraFrame *This,
    /* [in] */ LONG a_vlIndex,
    /* [retval][out] */ INPObject **ppObject);

void __RPC_STUB INPCameraFrame_Item_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_get_Id_Proxy(
    INPCameraFrame *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCameraFrame_get_Id_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_get_SwitchState_Proxy(
    INPCameraFrame *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPCameraFrame_get_SwitchState_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_get_IsEmpty_Proxy(
    INPCameraFrame *This,
    /* [retval][out] */ VARIANT_BOOL *pVal);

void __RPC_STUB INPCameraFrame_get_IsEmpty_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_get_IsCorrupt_Proxy(
    INPCameraFrame *This,
    /* [retval][out] */ VARIANT_BOOL *pVal);

void __RPC_STUB INPCameraFrame_get_IsCorrupt_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_get_IsGreyscale_Proxy(
    INPCameraFrame *This,
    /* [retval][out] */ VARIANT_BOOL *pVal);

void __RPC_STUB INPCameraFrame_get_IsGreyscale_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_get_TimeStamp_Proxy(
    INPCameraFrame *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPCameraFrame_get_TimeStamp_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_get_TimeStampFrequency_Proxy(
    INPCameraFrame *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPCameraFrame_get_TimeStampFrequency_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_GetObjectData_Proxy(
    INPCameraFrame *This,
    /* [in] */ BYTE *Buffer,
    /* [in] */ INT BufferSize,
    /* [retval][out] */ LONG *pObjectCount);

void __RPC_STUB INPCameraFrame_GetObjectData_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPCameraFrame_Free_Proxy(
    INPCameraFrame *This);

void __RPC_STUB INPCameraFrame_Free_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

#endif /* __INPCameraFrame_INTERFACE_DEFINED__ */

#ifndef __INPObject_INTERFACE_DEFINED__
#define __INPObject_INTERFACE_DEFINED__

/* interface INPObject */
/* [unique][helpstring][dual][uuid][object] */

EXTERN_C const IID IID_INPObject;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("6E439CE4-AB0D-44B8-BF1E-644C5CC489DC")
INPObject:
public
IDispatch
{
public:
    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Area(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_X(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Y(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Score(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Rank(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Width(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Height(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Transform(
        /* [in] */ INPCamera * pCamera) = 0;
};

#else /* C style interface */

typedef struct INPObjectVtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        INPObject *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        INPObject *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        INPObject *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        INPObject *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        INPObject *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        INPObject *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        INPObject *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Area)(
        INPObject *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_X)(
        INPObject *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Y)(
        INPObject *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Score)(
        INPObject *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Rank)(
        INPObject *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Width)(
        INPObject *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Height)(
        INPObject *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Transform)(
        INPObject *This,
        /* [in] */ INPCamera *pCamera);

    END_INTERFACE
} INPObjectVtbl;

interface INPObject
{
    CONST_VTBL struct INPObjectVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define INPObject_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define INPObject_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define INPObject_Release(This) \
    (This)->lpVtbl->Release(This)

#define INPObject_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define INPObject_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define INPObject_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define INPObject_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#define INPObject_get_Area(This, pVal) \
    (This)->lpVtbl->get_Area(This, pVal)

#define INPObject_get_X(This, pVal) \
    (This)->lpVtbl->get_X(This, pVal)

#define INPObject_get_Y(This, pVal) \
    (This)->lpVtbl->get_Y(This, pVal)

#define INPObject_get_Score(This, pVal) \
    (This)->lpVtbl->get_Score(This, pVal)

#define INPObject_get_Rank(This, pVal) \
    (This)->lpVtbl->get_Rank(This, pVal)

#define INPObject_get_Width(This, pVal) \
    (This)->lpVtbl->get_Width(This, pVal)

#define INPObject_get_Height(This, pVal) \
    (This)->lpVtbl->get_Height(This, pVal)

#define INPObject_Transform(This, pCamera) \
    (This)->lpVtbl->Transform(This, pCamera)

#endif /* COBJMACROS */

#endif /* C style interface */

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPObject_get_Area_Proxy(
    INPObject *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPObject_get_Area_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPObject_get_X_Proxy(
    INPObject *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPObject_get_X_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPObject_get_Y_Proxy(
    INPObject *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPObject_get_Y_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPObject_get_Score_Proxy(
    INPObject *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPObject_get_Score_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPObject_get_Rank_Proxy(
    INPObject *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPObject_get_Rank_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPObject_get_Width_Proxy(
    INPObject *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPObject_get_Width_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPObject_get_Height_Proxy(
    INPObject *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPObject_get_Height_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPObject_Transform_Proxy(
    INPObject *This,
    /* [in] */ INPCamera *pCamera);

void __RPC_STUB INPObject_Transform_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

#endif /* __INPObject_INTERFACE_DEFINED__ */

#ifndef __INPSmoothing_INTERFACE_DEFINED__
#define __INPSmoothing_INTERFACE_DEFINED__

/* interface INPSmoothing */
/* [unique][helpstring][dual][uuid][object] */

EXTERN_C const IID IID_INPSmoothing;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("0EDD3505-855C-4D91-A9C1-DCBEC1B816FA")
INPSmoothing:
public
IDispatch
{
public:
    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Amount(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_Amount(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_X(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Y(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Update(
        /* [in] */ VARIANT ValX,
        VARIANT ValY) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Reset(void) = 0;
};

#else /* C style interface */

typedef struct INPSmoothingVtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        INPSmoothing *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        INPSmoothing *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        INPSmoothing *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        INPSmoothing *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        INPSmoothing *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        INPSmoothing *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        INPSmoothing *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Amount)(
        INPSmoothing *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_Amount)(
        INPSmoothing *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_X)(
        INPSmoothing *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Y)(
        INPSmoothing *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Update)(
        INPSmoothing *This,
        /* [in] */ VARIANT ValX,
        VARIANT ValY);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Reset)(
        INPSmoothing *This);

    END_INTERFACE
} INPSmoothingVtbl;

interface INPSmoothing
{
    CONST_VTBL struct INPSmoothingVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define INPSmoothing_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define INPSmoothing_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define INPSmoothing_Release(This) \
    (This)->lpVtbl->Release(This)

#define INPSmoothing_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define INPSmoothing_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define INPSmoothing_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define INPSmoothing_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#define INPSmoothing_get_Amount(This, pVal) \
    (This)->lpVtbl->get_Amount(This, pVal)

#define INPSmoothing_put_Amount(This, Val) \
    (This)->lpVtbl->put_Amount(This, Val)

#define INPSmoothing_get_X(This, pVal) \
    (This)->lpVtbl->get_X(This, pVal)

#define INPSmoothing_get_Y(This, pVal) \
    (This)->lpVtbl->get_Y(This, pVal)

#define INPSmoothing_Update(This, ValX, ValY) \
    (This)->lpVtbl->Update(This, ValX, ValY)

#define INPSmoothing_Reset(This) \
    (This)->lpVtbl->Reset(This)

#endif /* COBJMACROS */

#endif /* C style interface */

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPSmoothing_get_Amount_Proxy(
    INPSmoothing *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPSmoothing_get_Amount_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPSmoothing_put_Amount_Proxy(
    INPSmoothing *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPSmoothing_put_Amount_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPSmoothing_get_X_Proxy(
    INPSmoothing *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPSmoothing_get_X_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPSmoothing_get_Y_Proxy(
    INPSmoothing *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPSmoothing_get_Y_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPSmoothing_Update_Proxy(
    INPSmoothing *This,
    /* [in] */ VARIANT ValX,
    VARIANT ValY);

void __RPC_STUB INPSmoothing_Update_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPSmoothing_Reset_Proxy(
    INPSmoothing *This);

void __RPC_STUB INPSmoothing_Reset_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

#endif /* __INPSmoothing_INTERFACE_DEFINED__ */

#ifndef __INPVector_INTERFACE_DEFINED__
#define __INPVector_INTERFACE_DEFINED__

/* interface INPVector */
/* [unique][helpstring][dual][uuid][object] */

EXTERN_C const IID IID_INPVector;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("9124C9A8-9296-4E89-973D-4F3C502E36CA")
INPVector:
public
IDispatch
{
public:
    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Yaw(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Pitch(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Roll(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_X(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Y(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Z(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Update(
        /* [in] */ INPCamera * pCamera,
        /* [in] */ INPCameraFrame * pFrame) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Reset(void) = 0;
};

#else /* C style interface */

typedef struct INPVectorVtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        INPVector *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        INPVector *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        INPVector *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        INPVector *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        INPVector *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        INPVector *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        INPVector *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Yaw)(
        INPVector *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Pitch)(
        INPVector *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Roll)(
        INPVector *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_X)(
        INPVector *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Y)(
        INPVector *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Z)(
        INPVector *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Update)(
        INPVector *This,
        /* [in] */ INPCamera *pCamera,
        /* [in] */ INPCameraFrame *pFrame);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Reset)(
        INPVector *This);

    END_INTERFACE
} INPVectorVtbl;

interface INPVector
{
    CONST_VTBL struct INPVectorVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define INPVector_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define INPVector_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define INPVector_Release(This) \
    (This)->lpVtbl->Release(This)

#define INPVector_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define INPVector_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define INPVector_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define INPVector_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#define INPVector_get_Yaw(This, pVal) \
    (This)->lpVtbl->get_Yaw(This, pVal)

#define INPVector_get_Pitch(This, pVal) \
    (This)->lpVtbl->get_Pitch(This, pVal)

#define INPVector_get_Roll(This, pVal) \
    (This)->lpVtbl->get_Roll(This, pVal)

#define INPVector_get_X(This, pVal) \
    (This)->lpVtbl->get_X(This, pVal)

#define INPVector_get_Y(This, pVal) \
    (This)->lpVtbl->get_Y(This, pVal)

#define INPVector_get_Z(This, pVal) \
    (This)->lpVtbl->get_Z(This, pVal)

#define INPVector_Update(This, pCamera, pFrame) \
    (This)->lpVtbl->Update(This, pCamera, pFrame)

#define INPVector_Reset(This) \
    (This)->lpVtbl->Reset(This)

#endif /* COBJMACROS */

#endif /* C style interface */

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector_get_Yaw_Proxy(
    INPVector *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector_get_Yaw_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector_get_Pitch_Proxy(
    INPVector *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector_get_Pitch_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector_get_Roll_Proxy(
    INPVector *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector_get_Roll_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector_get_X_Proxy(
    INPVector *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector_get_X_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector_get_Y_Proxy(
    INPVector *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector_get_Y_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector_get_Z_Proxy(
    INPVector *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector_get_Z_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPVector_Update_Proxy(
    INPVector *This,
    /* [in] */ INPCamera *pCamera,
    /* [in] */ INPCameraFrame *pFrame);

void __RPC_STUB INPVector_Update_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPVector_Reset_Proxy(
    INPVector *This);

void __RPC_STUB INPVector_Reset_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

#endif /* __INPVector_INTERFACE_DEFINED__ */

#ifndef __INPPoint_INTERFACE_DEFINED__
#define __INPPoint_INTERFACE_DEFINED__

/* interface INPPoint */
/* [unique][helpstring][dual][uuid][object] */

EXTERN_C const IID IID_INPPoint;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("9124C9F0-9296-4E89-973D-4F3C502E36CA")
INPPoint:
public
IDispatch
{
public:
    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_X(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Y(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Z(
        /* [retval][out] */ VARIANT * pVal) = 0;
};

#else /* C style interface */

typedef struct INPPointVtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        INPPoint *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        INPPoint *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        INPPoint *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        INPPoint *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        INPPoint *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        INPPoint *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        INPPoint *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_X)(
        INPPoint *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Y)(
        INPPoint *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Z)(
        INPPoint *This,
        /* [retval][out] */ VARIANT *pVal);

    END_INTERFACE
} INPPointVtbl;

interface INPPoint
{
    CONST_VTBL struct INPPointVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define INPPoint_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define INPPoint_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define INPPoint_Release(This) \
    (This)->lpVtbl->Release(This)

#define INPPoint_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define INPPoint_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define INPPoint_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define INPPoint_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#define INPPoint_get_X(This, pVal) \
    (This)->lpVtbl->get_X(This, pVal)

#define INPPoint_get_Y(This, pVal) \
    (This)->lpVtbl->get_Y(This, pVal)

#define INPPoint_get_Z(This, pVal) \
    (This)->lpVtbl->get_Z(This, pVal)

#endif /* COBJMACROS */

#endif /* C style interface */

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPPoint_get_X_Proxy(
    INPPoint *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPPoint_get_X_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPPoint_get_Y_Proxy(
    INPPoint *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPPoint_get_Y_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPPoint_get_Z_Proxy(
    INPPoint *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPPoint_get_Z_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

#endif /* __INPPoint_INTERFACE_DEFINED__ */

#ifndef __INPVector2_INTERFACE_DEFINED__
#define __INPVector2_INTERFACE_DEFINED__

/* interface INPVector2 */
/* [unique][helpstring][dual][uuid][object] */

EXTERN_C const IID IID_INPVector2;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("9124C9A9-9296-4E89-973D-4F3C502E36CA")
INPVector2:
public
IDispatch
{
public:
    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Yaw(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Pitch(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Roll(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_X(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Y(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Z(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_dist01(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_dist01(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_dist02(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_dist02(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_dist12(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_dist12(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_distol(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_distol(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Tracking(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Update(
        /* [in] */ INPCamera * pCamera,
        /* [in] */ INPCameraFrame * pFrame) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Reset(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE GetPoint(
        /* [in] */ int nPoint,
        /* [retval][out] */ INPPoint **ppPoint) = 0;
};

#else /* C style interface */

typedef struct INPVector2Vtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        INPVector2 *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        INPVector2 *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        INPVector2 *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        INPVector2 *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        INPVector2 *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        INPVector2 *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        INPVector2 *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Yaw)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Pitch)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Roll)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_X)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Y)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Z)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_dist01)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_dist01)(
        INPVector2 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_dist02)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_dist02)(
        INPVector2 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_dist12)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_dist12)(
        INPVector2 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_distol)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_distol)(
        INPVector2 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Tracking)(
        INPVector2 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Update)(
        INPVector2 *This,
        /* [in] */ INPCamera *pCamera,
        /* [in] */ INPCameraFrame *pFrame);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Reset)(
        INPVector2 *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *GetPoint)(
        INPVector2 *This,
        /* [in] */ int nPoint,
        /* [retval][out] */ INPPoint **ppPoint);

    END_INTERFACE
} INPVector2Vtbl;

interface INPVector2
{
    CONST_VTBL struct INPVector2Vtbl *lpVtbl;
};

#ifdef COBJMACROS

#define INPVector2_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define INPVector2_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define INPVector2_Release(This) \
    (This)->lpVtbl->Release(This)

#define INPVector2_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define INPVector2_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define INPVector2_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define INPVector2_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#define INPVector2_get_Yaw(This, pVal) \
    (This)->lpVtbl->get_Yaw(This, pVal)

#define INPVector2_get_Pitch(This, pVal) \
    (This)->lpVtbl->get_Pitch(This, pVal)

#define INPVector2_get_Roll(This, pVal) \
    (This)->lpVtbl->get_Roll(This, pVal)

#define INPVector2_get_X(This, pVal) \
    (This)->lpVtbl->get_X(This, pVal)

#define INPVector2_get_Y(This, pVal) \
    (This)->lpVtbl->get_Y(This, pVal)

#define INPVector2_get_Z(This, pVal) \
    (This)->lpVtbl->get_Z(This, pVal)

#define INPVector2_get_dist01(This, pVal) \
    (This)->lpVtbl->get_dist01(This, pVal)

#define INPVector2_put_dist01(This, Val) \
    (This)->lpVtbl->put_dist01(This, Val)

#define INPVector2_get_dist02(This, pVal) \
    (This)->lpVtbl->get_dist02(This, pVal)

#define INPVector2_put_dist02(This, Val) \
    (This)->lpVtbl->put_dist02(This, Val)

#define INPVector2_get_dist12(This, pVal) \
    (This)->lpVtbl->get_dist12(This, pVal)

#define INPVector2_put_dist12(This, Val) \
    (This)->lpVtbl->put_dist12(This, Val)

#define INPVector2_get_distol(This, pVal) \
    (This)->lpVtbl->get_distol(This, pVal)

#define INPVector2_put_distol(This, Val) \
    (This)->lpVtbl->put_distol(This, Val)

#define INPVector2_get_Tracking(This, pVal) \
    (This)->lpVtbl->get_Tracking(This, pVal)

#define INPVector2_Update(This, pCamera, pFrame) \
    (This)->lpVtbl->Update(This, pCamera, pFrame)

#define INPVector2_Reset(This) \
    (This)->lpVtbl->Reset(This)

#define INPVector2_GetPoint(This, nPoint, ppPoint) \
    (This)->lpVtbl->GetPoint(This, nPoint, ppPoint)

#endif /* COBJMACROS */

#endif /* C style interface */

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_Yaw_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_Yaw_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_Pitch_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_Pitch_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_Roll_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_Roll_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_X_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_X_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_Y_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_Y_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_Z_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_Z_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_dist01_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_dist01_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector2_put_dist01_Proxy(
    INPVector2 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector2_put_dist01_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_dist02_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_dist02_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector2_put_dist02_Proxy(
    INPVector2 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector2_put_dist02_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_dist12_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_dist12_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector2_put_dist12_Proxy(
    INPVector2 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector2_put_dist12_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_distol_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_distol_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector2_put_distol_Proxy(
    INPVector2 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector2_put_distol_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector2_get_Tracking_Proxy(
    INPVector2 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector2_get_Tracking_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPVector2_Update_Proxy(
    INPVector2 *This,
    /* [in] */ INPCamera *pCamera,
    /* [in] */ INPCameraFrame *pFrame);

void __RPC_STUB INPVector2_Update_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPVector2_Reset_Proxy(
    INPVector2 *This);

void __RPC_STUB INPVector2_Reset_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPVector2_GetPoint_Proxy(
    INPVector2 *This,
    /* [in] */ int nPoint,
    /* [retval][out] */ INPPoint **ppPoint);

void __RPC_STUB INPVector2_GetPoint_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

#endif /* __INPVector2_INTERFACE_DEFINED__ */

#ifndef __INPVector3_INTERFACE_DEFINED__
#define __INPVector3_INTERFACE_DEFINED__

/* interface INPVector3 */
/* [unique][helpstring][dual][uuid][object] */

EXTERN_C const IID IID_INPVector3;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("9124C9AA-9296-4E89-973D-4F3C502E36CA")
INPVector3:
public
IDispatch
{
public:
    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Yaw(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Pitch(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Roll(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_X(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Y(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Z(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_dist01(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_dist01(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_dist02(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_dist02(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_dist12(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_dist12(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_distol(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_distol(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_Tracking(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_imagerPixelWidth(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_imagerPixelWidth(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_imagerPixelHeight(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_imagerPixelHeight(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_imagerMMWidth(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_imagerMMWidth(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_imagerMMHeight(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_imagerMMHeight(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_imagerMMFocalLength(
        /* [retval][out] */ VARIANT * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_imagerMMFocalLength(
        /* [in] */ VARIANT Val) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Update(
        /* [in] */ INPCamera * pCamera,
        /* [in] */ INPCameraFrame * pFrame) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Reset(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE GetPoint(
        /* [in] */ int nPoint,
        /* [retval][out] */ INPPoint **ppPoint) = 0;
};

#else /* C style interface */

typedef struct INPVector3Vtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        INPVector3 *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        INPVector3 *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        INPVector3 *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        INPVector3 *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        INPVector3 *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        INPVector3 *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        INPVector3 *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Yaw)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Pitch)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Roll)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_X)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Y)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Z)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_dist01)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_dist01)(
        INPVector3 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_dist02)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_dist02)(
        INPVector3 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_dist12)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_dist12)(
        INPVector3 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_distol)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_distol)(
        INPVector3 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_Tracking)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_imagerPixelWidth)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_imagerPixelWidth)(
        INPVector3 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_imagerPixelHeight)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_imagerPixelHeight)(
        INPVector3 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_imagerMMWidth)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_imagerMMWidth)(
        INPVector3 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_imagerMMHeight)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_imagerMMHeight)(
        INPVector3 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_imagerMMFocalLength)(
        INPVector3 *This,
        /* [retval][out] */ VARIANT *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_imagerMMFocalLength)(
        INPVector3 *This,
        /* [in] */ VARIANT Val);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Update)(
        INPVector3 *This,
        /* [in] */ INPCamera *pCamera,
        /* [in] */ INPCameraFrame *pFrame);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Reset)(
        INPVector3 *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *GetPoint)(
        INPVector3 *This,
        /* [in] */ int nPoint,
        /* [retval][out] */ INPPoint **ppPoint);

    END_INTERFACE
} INPVector3Vtbl;

interface INPVector3
{
    CONST_VTBL struct INPVector3Vtbl *lpVtbl;
};

#ifdef COBJMACROS

#define INPVector3_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define INPVector3_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define INPVector3_Release(This) \
    (This)->lpVtbl->Release(This)

#define INPVector3_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define INPVector3_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define INPVector3_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define INPVector3_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#define INPVector3_get_Yaw(This, pVal) \
    (This)->lpVtbl->get_Yaw(This, pVal)

#define INPVector3_get_Pitch(This, pVal) \
    (This)->lpVtbl->get_Pitch(This, pVal)

#define INPVector3_get_Roll(This, pVal) \
    (This)->lpVtbl->get_Roll(This, pVal)

#define INPVector3_get_X(This, pVal) \
    (This)->lpVtbl->get_X(This, pVal)

#define INPVector3_get_Y(This, pVal) \
    (This)->lpVtbl->get_Y(This, pVal)

#define INPVector3_get_Z(This, pVal) \
    (This)->lpVtbl->get_Z(This, pVal)

#define INPVector3_get_dist01(This, pVal) \
    (This)->lpVtbl->get_dist01(This, pVal)

#define INPVector3_put_dist01(This, Val) \
    (This)->lpVtbl->put_dist01(This, Val)

#define INPVector3_get_dist02(This, pVal) \
    (This)->lpVtbl->get_dist02(This, pVal)

#define INPVector3_put_dist02(This, Val) \
    (This)->lpVtbl->put_dist02(This, Val)

#define INPVector3_get_dist12(This, pVal) \
    (This)->lpVtbl->get_dist12(This, pVal)

#define INPVector3_put_dist12(This, Val) \
    (This)->lpVtbl->put_dist12(This, Val)

#define INPVector3_get_distol(This, pVal) \
    (This)->lpVtbl->get_distol(This, pVal)

#define INPVector3_put_distol(This, Val) \
    (This)->lpVtbl->put_distol(This, Val)

#define INPVector3_get_Tracking(This, pVal) \
    (This)->lpVtbl->get_Tracking(This, pVal)

#define INPVector3_get_imagerPixelWidth(This, pVal) \
    (This)->lpVtbl->get_imagerPixelWidth(This, pVal)

#define INPVector3_put_imagerPixelWidth(This, Val) \
    (This)->lpVtbl->put_imagerPixelWidth(This, Val)

#define INPVector3_get_imagerPixelHeight(This, pVal) \
    (This)->lpVtbl->get_imagerPixelHeight(This, pVal)

#define INPVector3_put_imagerPixelHeight(This, Val) \
    (This)->lpVtbl->put_imagerPixelHeight(This, Val)

#define INPVector3_get_imagerMMWidth(This, pVal) \
    (This)->lpVtbl->get_imagerMMWidth(This, pVal)

#define INPVector3_put_imagerMMWidth(This, Val) \
    (This)->lpVtbl->put_imagerMMWidth(This, Val)

#define INPVector3_get_imagerMMHeight(This, pVal) \
    (This)->lpVtbl->get_imagerMMHeight(This, pVal)

#define INPVector3_put_imagerMMHeight(This, Val) \
    (This)->lpVtbl->put_imagerMMHeight(This, Val)

#define INPVector3_get_imagerMMFocalLength(This, pVal) \
    (This)->lpVtbl->get_imagerMMFocalLength(This, pVal)

#define INPVector3_put_imagerMMFocalLength(This, Val) \
    (This)->lpVtbl->put_imagerMMFocalLength(This, Val)

#define INPVector3_Update(This, pCamera, pFrame) \
    (This)->lpVtbl->Update(This, pCamera, pFrame)

#define INPVector3_Reset(This) \
    (This)->lpVtbl->Reset(This)

#define INPVector3_GetPoint(This, nPoint, ppPoint) \
    (This)->lpVtbl->GetPoint(This, nPoint, ppPoint)

#endif /* COBJMACROS */

#endif /* C style interface */

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_Yaw_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_Yaw_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_Pitch_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_Pitch_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_Roll_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_Roll_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_X_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_X_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_Y_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_Y_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_Z_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_Z_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_dist01_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_dist01_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector3_put_dist01_Proxy(
    INPVector3 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector3_put_dist01_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_dist02_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_dist02_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector3_put_dist02_Proxy(
    INPVector3 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector3_put_dist02_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_dist12_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_dist12_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector3_put_dist12_Proxy(
    INPVector3 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector3_put_dist12_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_distol_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_distol_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector3_put_distol_Proxy(
    INPVector3 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector3_put_distol_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_Tracking_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_Tracking_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_imagerPixelWidth_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_imagerPixelWidth_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector3_put_imagerPixelWidth_Proxy(
    INPVector3 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector3_put_imagerPixelWidth_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_imagerPixelHeight_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_imagerPixelHeight_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector3_put_imagerPixelHeight_Proxy(
    INPVector3 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector3_put_imagerPixelHeight_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_imagerMMWidth_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_imagerMMWidth_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector3_put_imagerMMWidth_Proxy(
    INPVector3 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector3_put_imagerMMWidth_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_imagerMMHeight_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_imagerMMHeight_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector3_put_imagerMMHeight_Proxy(
    INPVector3 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector3_put_imagerMMHeight_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPVector3_get_imagerMMFocalLength_Proxy(
    INPVector3 *This,
    /* [retval][out] */ VARIANT *pVal);

void __RPC_STUB INPVector3_get_imagerMMFocalLength_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPVector3_put_imagerMMFocalLength_Proxy(
    INPVector3 *This,
    /* [in] */ VARIANT Val);

void __RPC_STUB INPVector3_put_imagerMMFocalLength_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPVector3_Update_Proxy(
    INPVector3 *This,
    /* [in] */ INPCamera *pCamera,
    /* [in] */ INPCameraFrame *pFrame);

void __RPC_STUB INPVector3_Update_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPVector3_Reset_Proxy(
    INPVector3 *This);

void __RPC_STUB INPVector3_Reset_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPVector3_GetPoint_Proxy(
    INPVector3 *This,
    /* [in] */ int nPoint,
    /* [retval][out] */ INPPoint **ppPoint);

void __RPC_STUB INPVector3_GetPoint_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

#endif /* __INPVector3_INTERFACE_DEFINED__ */

#ifndef __INPAvi_INTERFACE_DEFINED__
#define __INPAvi_INTERFACE_DEFINED__

/* interface INPAvi */
/* [unique][helpstring][dual][uuid][object] */

EXTERN_C const IID IID_INPAvi;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("9124CA00-9296-4E89-973D-4F3C502E36CA")
INPAvi:
public
IDispatch
{
public:
    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_FileName(
        /* [retval][out] */ BSTR * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_FileName(
        /* [in] */ BSTR Val) = 0;

    virtual /* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE get_FrameRate(
        /* [retval][out] */ LONG * pVal) = 0;

    virtual /* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE put_FrameRate(
        /* [in] */ LONG Val) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Start(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE Stop(void) = 0;

    virtual /* [helpstring][id] */ HRESULT STDMETHODCALLTYPE AddFrame(
        /* [in] */ INPCamera * pCamera,
        /* [in] */ INPCameraFrame * pFrame) = 0;
};

#else /* C style interface */

typedef struct INPAviVtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        INPAvi *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        INPAvi *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        INPAvi *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        INPAvi *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        INPAvi *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        INPAvi *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        INPAvi *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_FileName)(
        INPAvi *This,
        /* [retval][out] */ BSTR *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_FileName)(
        INPAvi *This,
        /* [in] */ BSTR Val);

    /* [helpstring][id][propget] */ HRESULT(STDMETHODCALLTYPE *get_FrameRate)(
        INPAvi *This,
        /* [retval][out] */ LONG *pVal);

    /* [helpstring][id][propput] */ HRESULT(STDMETHODCALLTYPE *put_FrameRate)(
        INPAvi *This,
        /* [in] */ LONG Val);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Start)(
        INPAvi *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *Stop)(
        INPAvi *This);

    /* [helpstring][id] */ HRESULT(STDMETHODCALLTYPE *AddFrame)(
        INPAvi *This,
        /* [in] */ INPCamera *pCamera,
        /* [in] */ INPCameraFrame *pFrame);

    END_INTERFACE
} INPAviVtbl;

interface INPAvi
{
    CONST_VTBL struct INPAviVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define INPAvi_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define INPAvi_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define INPAvi_Release(This) \
    (This)->lpVtbl->Release(This)

#define INPAvi_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define INPAvi_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define INPAvi_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define INPAvi_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#define INPAvi_get_FileName(This, pVal) \
    (This)->lpVtbl->get_FileName(This, pVal)

#define INPAvi_put_FileName(This, Val) \
    (This)->lpVtbl->put_FileName(This, Val)

#define INPAvi_get_FrameRate(This, pVal) \
    (This)->lpVtbl->get_FrameRate(This, pVal)

#define INPAvi_put_FrameRate(This, Val) \
    (This)->lpVtbl->put_FrameRate(This, Val)

#define INPAvi_Start(This) \
    (This)->lpVtbl->Start(This)

#define INPAvi_Stop(This) \
    (This)->lpVtbl->Stop(This)

#define INPAvi_AddFrame(This, pCamera, pFrame) \
    (This)->lpVtbl->AddFrame(This, pCamera, pFrame)

#endif /* COBJMACROS */

#endif /* C style interface */

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPAvi_get_FileName_Proxy(
    INPAvi *This,
    /* [retval][out] */ BSTR *pVal);

void __RPC_STUB INPAvi_get_FileName_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPAvi_put_FileName_Proxy(
    INPAvi *This,
    /* [in] */ BSTR Val);

void __RPC_STUB INPAvi_put_FileName_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propget] */ HRESULT STDMETHODCALLTYPE INPAvi_get_FrameRate_Proxy(
    INPAvi *This,
    /* [retval][out] */ LONG *pVal);

void __RPC_STUB INPAvi_get_FrameRate_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id][propput] */ HRESULT STDMETHODCALLTYPE INPAvi_put_FrameRate_Proxy(
    INPAvi *This,
    /* [in] */ LONG Val);

void __RPC_STUB INPAvi_put_FrameRate_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPAvi_Start_Proxy(
    INPAvi *This);

void __RPC_STUB INPAvi_Start_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPAvi_Stop_Proxy(
    INPAvi *This);

void __RPC_STUB INPAvi_Stop_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

/* [helpstring][id] */ HRESULT STDMETHODCALLTYPE INPAvi_AddFrame_Proxy(
    INPAvi *This,
    /* [in] */ INPCamera *pCamera,
    /* [in] */ INPCameraFrame *pFrame);

void __RPC_STUB INPAvi_AddFrame_Stub(
    IRpcStubBuffer *This,
    IRpcChannelBuffer *_pRpcChannelBuffer,
    PRPC_MESSAGE _pRpcMessage,
    DWORD *_pdwStubPhase);

#endif /* __INPAvi_INTERFACE_DEFINED__ */

#ifndef __OptiTrack_LIBRARY_DEFINED__
#define __OptiTrack_LIBRARY_DEFINED__

/* library OptiTrack */
/* [helpstring][version][uuid] */

EXTERN_C const IID LIBID_OptiTrack;

#ifndef ___INPCameraCollectionEvents_DISPINTERFACE_DEFINED__
#define ___INPCameraCollectionEvents_DISPINTERFACE_DEFINED__

/* dispinterface _INPCameraCollectionEvents */
/* [helpstring][uuid] */

EXTERN_C const IID DIID__INPCameraCollectionEvents;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("076F9DDA-1422-4B4D-926A-961DF5725B5A")
_INPCameraCollectionEvents:
public
IDispatch{};

#else /* C style interface */

typedef struct _INPCameraCollectionEventsVtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        _INPCameraCollectionEvents *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        _INPCameraCollectionEvents *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        _INPCameraCollectionEvents *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        _INPCameraCollectionEvents *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        _INPCameraCollectionEvents *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        _INPCameraCollectionEvents *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        _INPCameraCollectionEvents *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    END_INTERFACE
} _INPCameraCollectionEventsVtbl;

interface _INPCameraCollectionEvents
{
    CONST_VTBL struct _INPCameraCollectionEventsVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define _INPCameraCollectionEvents_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define _INPCameraCollectionEvents_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define _INPCameraCollectionEvents_Release(This) \
    (This)->lpVtbl->Release(This)

#define _INPCameraCollectionEvents_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define _INPCameraCollectionEvents_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define _INPCameraCollectionEvents_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define _INPCameraCollectionEvents_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#endif /* COBJMACROS */

#endif /* C style interface */

#endif /* ___INPCameraCollectionEvents_DISPINTERFACE_DEFINED__ */

#ifndef ___INPCameraEvents_DISPINTERFACE_DEFINED__
#define ___INPCameraEvents_DISPINTERFACE_DEFINED__

/* dispinterface _INPCameraEvents */
/* [helpstring][uuid] */

EXTERN_C const IID DIID__INPCameraEvents;

#if defined(__cplusplus) && !defined(CINTERFACE)

MIDL_INTERFACE("A50B57C5-7472-4F16-BC14-2345B8D24BFD")
_INPCameraEvents:
public
IDispatch{};

#else /* C style interface */

typedef struct _INPCameraEventsVtbl
{
    BEGIN_INTERFACE

    HRESULT(STDMETHODCALLTYPE *QueryInterface)(
        _INPCameraEvents *This,
        /* [in] */ REFIID riid,
        /* [iid_is][out] */ void **ppvObject);

    ULONG(STDMETHODCALLTYPE *AddRef)(
        _INPCameraEvents *This);

    ULONG(STDMETHODCALLTYPE *Release)(
        _INPCameraEvents *This);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfoCount)(
        _INPCameraEvents *This,
        /* [out] */ UINT *pctinfo);

    HRESULT(STDMETHODCALLTYPE *GetTypeInfo)(
        _INPCameraEvents *This,
        /* [in] */ UINT iTInfo,
        /* [in] */ LCID lcid,
        /* [out] */ ITypeInfo **ppTInfo);

    HRESULT(STDMETHODCALLTYPE *GetIDsOfNames)(
        _INPCameraEvents *This,
        /* [in] */ REFIID riid,
        /* [size_is][in] */ LPOLESTR *rgszNames,
        /* [in] */ UINT cNames,
        /* [in] */ LCID lcid,
        /* [size_is][out] */ DISPID *rgDispId);

    /* [local] */ HRESULT(STDMETHODCALLTYPE *Invoke)(
        _INPCameraEvents *This,
        /* [in] */ DISPID dispIdMember,
        /* [in] */ REFIID riid,
        /* [in] */ LCID lcid,
        /* [in] */ WORD wFlags,
        /* [out][in] */ DISPPARAMS *pDispParams,
        /* [out] */ VARIANT *pVarResult,
        /* [out] */ EXCEPINFO *pExcepInfo,
        /* [out] */ UINT *puArgErr);

    END_INTERFACE
} _INPCameraEventsVtbl;

interface _INPCameraEvents
{
    CONST_VTBL struct _INPCameraEventsVtbl *lpVtbl;
};

#ifdef COBJMACROS

#define _INPCameraEvents_QueryInterface(This, riid, ppvObject) \
    (This)->lpVtbl->QueryInterface(This, riid, ppvObject)

#define _INPCameraEvents_AddRef(This) \
    (This)->lpVtbl->AddRef(This)

#define _INPCameraEvents_Release(This) \
    (This)->lpVtbl->Release(This)

#define _INPCameraEvents_GetTypeInfoCount(This, pctinfo) \
    (This)->lpVtbl->GetTypeInfoCount(This, pctinfo)

#define _INPCameraEvents_GetTypeInfo(This, iTInfo, lcid, ppTInfo) \
    (This)->lpVtbl->GetTypeInfo(This, iTInfo, lcid, ppTInfo)

#define _INPCameraEvents_GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId) \
    (This)->lpVtbl->GetIDsOfNames(This, riid, rgszNames, cNames, lcid, rgDispId)

#define _INPCameraEvents_Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr) \
    (This)->lpVtbl->Invoke(This, dispIdMember, riid, lcid, wFlags, pDispParams, pVarResult, pExcepInfo, puArgErr)

#endif /* COBJMACROS */

#endif /* C style interface */

#endif /* ___INPCameraEvents_DISPINTERFACE_DEFINED__ */

EXTERN_C const CLSID CLSID_NPCameraCollection;

#ifdef __cplusplus

class DECLSPEC_UUID("1CA83C6F-70A6-40EB-836F-D9EEC0BD168F")
    NPCameraCollection;
#endif

EXTERN_C const CLSID CLSID_NPCamera;

#ifdef __cplusplus

class DECLSPEC_UUID("77686C4C-8402-42CE-ADF2-913B53E0A25B")
    NPCamera;
#endif

EXTERN_C const CLSID CLSID_NPCameraFrame;

#ifdef __cplusplus

class DECLSPEC_UUID("4656500B-863B-48F6-8725-AB029769EA89")
    NPCameraFrame;
#endif

EXTERN_C const CLSID CLSID_NPObject;

#ifdef __cplusplus

class DECLSPEC_UUID("B696B174-5B53-4DDD-B78B-CA75C072C85A")
    NPObject;
#endif

EXTERN_C const CLSID CLSID_NPSmoothing;

#ifdef __cplusplus

class DECLSPEC_UUID("B4CA710D-9B17-42C3-846B-FC16876B6D5E")
    NPSmoothing;
#endif

EXTERN_C const CLSID CLSID_NPVector;

#ifdef __cplusplus

class DECLSPEC_UUID("FE7D5FB0-0560-49ED-BF49-CE9996C62A6B")
    NPVector;
#endif

EXTERN_C const CLSID CLSID_NPPoint;

#ifdef __cplusplus

class DECLSPEC_UUID("FE7D5FB2-0560-49ED-BF49-CE9996C62A6B")
    NPPoint;
#endif

EXTERN_C const CLSID CLSID_NPAvi;

#ifdef __cplusplus

class DECLSPEC_UUID("FE7D5FB3-0560-49ED-BF49-CE9996C62A6B")
    NPAvi;
#endif
#endif /* __OptiTrack_LIBRARY_DEFINED__ */

/* Additional Prototypes for ALL interfaces */

unsigned long __RPC_USER BSTR_UserSize(unsigned long *, unsigned long, BSTR *);
unsigned char *__RPC_USER BSTR_UserMarshal(unsigned long *, unsigned char *, BSTR *);
unsigned char *__RPC_USER BSTR_UserUnmarshal(unsigned long *, unsigned char *, BSTR *);
void __RPC_USER BSTR_UserFree(unsigned long *, BSTR *);

unsigned long __RPC_USER VARIANT_UserSize(unsigned long *, unsigned long, VARIANT *);
unsigned char *__RPC_USER VARIANT_UserMarshal(unsigned long *, unsigned char *, VARIANT *);
unsigned char *__RPC_USER VARIANT_UserUnmarshal(unsigned long *, unsigned char *, VARIANT *);
void __RPC_USER VARIANT_UserFree(unsigned long *, VARIANT *);

/* end of Additional Prototypes */

#ifdef __cplusplus
}
#endif

#endif
