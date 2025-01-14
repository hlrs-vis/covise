/* DTrackSDK in C++: DTrackDataTypes.hpp
 *
 * Data type definitions.
 *
 * Copyright (c) 2007-2023 Advanced Realtime Tracking GmbH & Co. KG
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _ART_DTRACKSDK_DATATYPES_HPP_
#define _ART_DTRACKSDK_DATATYPES_HPP_

#include <vector>

namespace DTrackSDK_Datatypes {

/**
 * \brief Quaternion.
 */
struct DTrackQuaternion
{
	double w;  //!< Quaternion component w
	double x;  //!< Quaternion component x
	double y;  //!< Quaternion component y
	double z;  //!< Quaternion component z
};

/**
 * \brief Helper to convert a rotation matrix into a quaternion.
 *
 * @param[in] rot Rotation matrix (column-wise)
 * @return        Quaternion
 */
DTrackQuaternion rot2quat( const double rot[ 9 ] );

// -----------------------------------------------------------------------------------------------------

/**
 * \brief Single marker data (3DOF).
 */
struct DTrackMarker
{
	int id;           //!< ID number (starting with 1)
	double quality;   //!< Quality (0.0 <= qu <= 1.0)
	double loc[ 3 ];  //!< Location (in [mm])
};

typedef DTrackMarker DTrack_Marker_Type_d;  //!< Alias for DTrackMarker. DEPRECATED.

// -----------------------------------------------------------------------------------------------------

/**
 * \brief Standard body data (6DOF).
 */
struct DTrackBody
{
	int id;              //!< ID number (starting with 0)
	double quality;      //!< Quality (0.0 <= qu <= 1.0, no tracking if -1.0)
	double loc[ 3 ];     //!< Location (in [mm])
	double rot[ 9 ];     //!< Rotation matrix (column-wise)
	double covref[ 3 ];  //!< Reference point of covariance (in [mm])
	double cov[ 36 ];    //!< 6x6-dimensional covariance matrix for the 6d pose (with 3d location in [mm], 3d euler angles in [rad]).

	/**
	 * \brief Returns if body is currently tracked.
	 *
	 * @return Is tracked?
	 */
	bool isTracked() const
	{ return ( quality >= 0.0 ); }

	/**
	 * \brief Returns rotation as quaternion.
	 *
	 * @return Quaternion
	 */
	DTrackQuaternion getQuaternion() const
	{ return rot2quat( rot ); }
};

typedef DTrackBody DTrack_Body_Type_d;  //!< Alias for DTrackBody. DEPRECATED.

// -----------------------------------------------------------------------------------------------------

/**
 * \brief Hybrid (optical-inertial) body data (6DOF).
 */
struct DTrackInertial
{
	int id;           //!< ID number (starting with 0)
	int st;           //!< State of hybrid body (0: not tracked, 1: inertial tracking, 2: optical tracking, 3: inertial and optical tracking)
	double error;     //!< Drift error estimate (only during inertial tracking, in [deg])
	double loc[ 3 ];  //!< Location (in [mm])
	double rot[ 9 ];  //!< Rotation matrix (column-wise)

	/**
	 * \brief Returns if body is currently tracked.
	 *
	 * @return Is tracked?
	 */
	bool isTracked() const
	{ return ( st > 0 ); }

	/**
	 * \brief Returns rotation as quaternion.
	 *
	 * @return Quaternion
	 */
	DTrackQuaternion getQuaternion() const
	{ return rot2quat( rot ); }
};

typedef DTrackInertial DTrack_Inertial_Type_d;  //!< Alias for DTrackInertial. DEPRECATED.

// -----------------------------------------------------------------------------------------------------

#define DTRACKSDK_FLYSTICK_MAX_BUTTON    16  //!< A.R.T. Flystick data: maximum number of buttons
#define DTRACKSDK_FLYSTICK_MAX_JOYSTICK   8  //!< A.R.T. Flystick data: maximum number of joystick values

/**
 * \brief A.R.T. Flystick data (6DOF + buttons).
 *
 * Note the maximum number of buttons and joystick values.
 */
struct DTrackFlyStick
{
	int id;            //!< ID number (starting with 0)
	double quality;    //!< Quality (0.0 <= qu <= 1.0, no tracking if -1.0)
	int num_button;    //!< Number of buttons
	int button[ DTRACKSDK_FLYSTICK_MAX_BUTTON ];  //!< Button state (1 pressed, 0 not pressed): 0 front, 1..n-1 right to left
	int num_joystick;  //!< Number of joystick values
	double joystick[ DTRACKSDK_FLYSTICK_MAX_JOYSTICK ];  //!< Joystick value (-1.0 <= joystick <= 1.0); 0 horizontal, 1 vertical
	double loc[ 3 ];   //!< Location (in [mm])
	double rot[ 9 ];   //!< Rotation matrix (column-wise)

	/**
	 * \brief Returns if Flystick is currently tracked.
	 *
	 * @return Is tracked?
	 */
	bool isTracked() const
	{ return ( quality >= 0.0 ); }

	/**
	 * \brief Returns rotation as quaternion.
	 *
	 * @return Quaternion
	 */
	DTrackQuaternion getQuaternion() const
	{ return rot2quat( rot ); }
};

typedef DTrackFlyStick DTrack_FlyStick_Type_d;  //!< Alias for DTrackFlyStick. DEPRECATED.

// -----------------------------------------------------------------------------------------------------

#define DTRACKSDK_MEATOOL_MAX_BUTTON  16  //!< Measurement tool data: maximum number of buttons

/**
 * \brief Measurement Tool data (6DOF + buttons).
 *
 * Note the maximum number of buttons.
 */
struct DTrackMeaTool
{
	int id;            //!< ID number (starting with 0)
	double quality;    //!< Quality (0.0 <= qu <= 1.0, no tracking if -1.0)
	int num_button;    //!< Number of buttons
	int button[ DTRACKSDK_MEATOOL_MAX_BUTTON ];  //!< Button state (1 pressed, 0 not pressed): 0 point measurement state
	double loc[ 3 ];   //!< Location (in [mm])
	double rot[ 9 ];   //!< Rotation matrix (column-wise)
	double tipradius;  //!< Radius of tip (in [mm]) if applicable
	double cov[ 9 ];   //!< Covariance of location (column-wise; in [mm^2])

	/**
	 * \brief Returns if Measurement Tool is currently tracked.
	 *
	 * @return Is tracked?
	 */
	bool isTracked() const
	{ return ( quality >= 0.0 ); }

	/**
	 * \brief Returns rotation as quaternion.
	 *
	 * @return Quaternion
	 */
	DTrackQuaternion getQuaternion() const
	{ return rot2quat( rot ); }
};

typedef DTrackMeaTool DTrack_MeaTool_Type_d;  //!< Alias for DTrackMeaTool. DEPRECATED.

// -----------------------------------------------------------------------------------------------------

/**
 * \brief Measurement Tool reference data (6DOF).
 */
struct DTrackMeaRef
{
	int id;           //!< ID number (starting with 0)
	double quality;   //!< Quality (0.0 <= qu <= 1.0, no tracking if -1.0)
	double loc[ 3 ];  //!< Location (in [mm])
	double rot[ 9 ];  //!< Rotation matrix (column-wise)

	/**
	 * \brief Returns if Measurement Tool reference is currently tracked.
	 *
	 * @return Is tracked?
	 */
	bool isTracked() const
	{ return ( quality >= 0.0 ); }

	/**
	 * \brief Returns rotation as quaternion.
	 *
	 * @return Quaternion
	 */
	DTrackQuaternion getQuaternion() const
	{ return rot2quat( rot ); }
};

typedef DTrackMeaRef DTrack_MeaRef_Type_d;  //!< Alias for DTrackMeaRef. DEPRECATED.

// -----------------------------------------------------------------------------------------------------

#define DTRACKSDK_HAND_MAX_FINGER    5  //!< Fingertracking hand data: maximum number of fingers

/**
 * \brief A.R.T. FINGERTRACKING hand data (6DOF + fingers).
 */
struct DTrackHand
{
	int id;           //!< ID number (starting with 0)
	double quality;   //!< Quality (0.0 <= qu <= 1.0, no tracking if -1.0)
	int lr;           //!< Left (0) or right (1) hand
	int nfinger;      //!< Number of fingers (maximum 5)
	double loc[ 3 ];  //!< Location of back of the hand (in [mm])
	double rot[ 9 ];  //!< Rotation matrix of back of the hand (column-wise)

	/**
	 * \brief Returns if hand is currently tracked.
	 *
	 * @return Is tracked?
	 */
	bool isTracked() const
	{ return ( quality >= 0.0 ); }

	/**
	 * \brief Returns rotation of back of the hand as quaternion.
	 *
	 * @return Quaternion
	 */
	DTrackQuaternion getQuaternion() const
	{ return rot2quat( rot ); }

	/**
	 * \brief A.R.T. FINGERTRACKING finger data.
	 */
	struct DTrackFinger
	{
		double loc[ 3 ];            //!< Location of tip (in [mm])
		double rot[ 9 ];            //!< Rotation matrix of outermost phalanx (column-wise)
		double radiustip;           //!< Radius of tip (in [mm])
		double lengthphalanx[ 3 ];  //!< Length of phalanxes (order: outermost, middle, innermost; in [mm])
		double anglephalanx[ 2 ];   //!< Angle between phalanxes (order: outermost, innermost; in [deg])

		/**
		 * \brief Returns rotation of outermost phalanx as quaternion.
		 *
		 * @return Quaternion
		 */
		DTrackQuaternion getQuaternion() const
		{ return rot2quat( rot ); }
	} finger[ DTRACKSDK_HAND_MAX_FINGER ];  //!< Finger data (order: thumb, index finger, middle finger, ...)
};

typedef DTrackHand DTrack_Hand_Type_d;  //!< Alias for DTrackHand. DEPRECATED.

// -----------------------------------------------------------------------------------------------------

#define DTRACKSDK_HUMAN_MAX_JOINTS 200  //!< ART-Human model: maximum number of joints

/**
 * \brief ART-Human model (joints (6DOF) including optional Fingertracking).
 *
 * Note the maximum number of joints.
 */
struct DTrackHuman
{
	int id;          //!< ID number of human model (starting with 0)
	int num_joints;  //!< Number of joints

	/**
	 * \brief Returns if human model is currently tracked.
	 *
	 * @return Is tracked?
	 */
	bool isTracked() const
	{ return ( num_joints > 0 ); }

	/**
	 * \brief ART-Human joint data.
	 */
	struct DTrackJoint
	{
		int id;           //!< ID number of joint (starting with 0)
		double quality;   //!< Quality of joint (0.0 <= qu <= 1.0, no tracking if -1.0)
		double loc[ 3 ];  //!< Location of joint (in [mm])
		double ang[ 3 ];  //!< Angles in relation to joint coordinate system; DEPRECATED
		double rot[ 9 ];  //!< Rotation matrix of joint (column-wise) in relation to room coordinate system

		/**
		 * \brief Returns if joint is currently tracked.
		 *
		 * @return Is tracked?
		 */
		bool isTracked() const
		{ return ( quality >= 0.0 ); }

		/**
		 * \brief Returns rotation of joint as quaternion.
		 *
		 * @return Quaternion
		 */
		DTrackQuaternion getQuaternion() const
		{ return rot2quat( rot ); }
	} joint[ DTRACKSDK_HUMAN_MAX_JOINTS ];  //!< Joint data
};

typedef DTrackHuman DTrack_Human_Type_d;  //!< Alias for DTrackHuman. DEPRECATED.

// -----------------------------------------------------------------------------------------------------

/**
 * \brief Camera status data.
 *
 * Note that this struct may be enhanced in future DTrackSDK versions.
 */
struct DTrackCameraStatus
{
	int idCamera;            //!< ID number of the camera (starting with 0)

	int numReflections;      //!< Number of 2DOF reflections seen by this camera
	int numReflectionsUsed;  //!< Number of seen 2DOF reflections used for 6DOF tracking
	int maxIntensity;        //!< Intensity of the brightest pixel (between 0 and 10)
};

/**
 * \brief System status data.
 *
 * Note that this struct may be enhanced in future DTrackSDK versions.
 */
struct DTrackStatus
{
	// general status values
	int numCameras;                //!< Number of cameras
	int numTrackedBodies;          //!< Number of currently tracked 6DOF bodies
	int numTrackedMarkers;         //!< Number of currently found additional 3DOF markers

	// message statistics
	int numCameraErrorMessages;    //!< Number of camera-related error messages (since booting)
	int numCameraWarningMessages;  //!< Number of camera-related warning messages (since booting)
	int numOtherErrorMessages;     //!< Number of other error messages (since booting)
	int numOtherWarningMessages;   //!< Number of other warning messages (since booting)
	int numInfoMessages;           //!< Number of info messages (since booting)

	// camera status values
	std::vector< DTrackCameraStatus > cameraStatus;  //!< Camera status
};


}  // namespace DTrackSDK_Datatypes

#endif  // _ART_DTRACKSDK_DATATYPES_HPP_

