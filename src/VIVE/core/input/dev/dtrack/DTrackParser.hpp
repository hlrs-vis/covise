/* DTrackSDK in C++: DTrackParser.hpp
 *
 * Functions to process DTrack UDP packets (ASCII protocol).
 *
 * Copyright (c) 2013-2022 Advanced Realtime Tracking GmbH & Co. KG
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
 * 
 * Purpose:
 *  - DTrack network protocol according to:
 *    'DTrack2 User Manual, Technical Appendix' or 'DTRACK3 Programmer's Guide'
 */

#ifndef _ART_DTRACKSDK_PARSER_HPP_
#define _ART_DTRACKSDK_PARSER_HPP_

#include "DTrackDataTypes.hpp"

#include <vector>

using namespace DTrackSDK_Datatypes;

/**
 * \brief DTrack Parser class.
 */
class DTrackParser
{
protected:

	/**
	 * \brief Constructor.
	 */
	DTrackParser();

	/**
	 * \brief Destructor.
	 */
	virtual ~DTrackParser();

	/**
	 * \brief Set default values at start of a new frame.
	 */
	void startFrame();

	/**
	 * \brief Final adjustments after processing all data for a frame.
	 */
	void endFrame();

	/**
	 * \brief Parses a single line of data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line One line of data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine( char **line );

public:

	/**
	 * \brief Get frame counter.
	 *
	 * Refers to last received frame.
	 *
	 * @return Frame counter
	 */
	unsigned int getFrameCounter() const;

	/**
	 * \brief Get timestamp.
	 *
	 * Refers to last received frame.
	 *
	 * @return Timestamp (-1 if information not available)
	 */
	double getTimeStamp() const;

	/**
	 * \brief Get number of calibrated standard bodies (as far as known).
	 *
	 * Refers to last received frame.
	 *
	 * @return Number of calibrated standard bodies
	 */
	int getNumBody() const;

	/**
	 * \brief Get standard body data.
	 *
	 * Refers to last received frame.
	 *
	 * @param[in] id Id, range 0 ..
	 * @return       Id-th standard body data; NULL in case of error
	 */
	const DTrackBody* getBody( int id ) const;

	/**
	 * \brief Get number of calibrated Flysticks.
	 *
	 * Refers to last received frame.
	 *
	 * @return Number of calibrated Flysticks
	 */
	int getNumFlyStick() const;

	/**
	 * \brief Get Flystick data.
	 *
	 * Refers to last received frame.
	 *
	 * @param[in] id Id, range 0 ..
	 * @return       Id-th Flystick data; NULL in case of error
	 */
	const DTrackFlyStick* getFlyStick( int id ) const;

	/**
	 * \brief Get number of calibrated Measurement Tools.
	 *
	 * Refers to last received frame.
	 *
	 * @return Number of calibrated Measurement Tools
	 */
	int getNumMeaTool() const;

	/**
	 * \brief Get Measurement Tool data.
	 *
	 * Refers to last received frame.
	 *
	 * @param[in] id Id, range 0 ..
	 * @return       Id-th Measurement Tool data; NULL in case of error
	 */
	const DTrackMeaTool* getMeaTool( int id ) const;

	/**
	 * \brief Get number of calibrated Measurement Tool references.
	 *
	 * Refers to last received frame.
	 *
	 * @return Number of calibrated Measurement Tool references
	 */
	int getNumMeaRef() const;

	/**
	 * \brief Get Measurement Tool reference data.
	 *
	 * Refers to last received frame.
	 *
	 * @param[in] id Id, range 0 ..
	 * @return       Id-th Measurement Tool reference data; NULL in case of error
	 */
	const DTrackMeaRef* getMeaRef( int id ) const;

	/**
	 * \brief Get number of calibrated A.R.T. FINGERTRACKING hands (as far as known).
	 *
	 * Refers to last received frame.
	 *
	 * @return Number of calibrated A.R.T. FINGERTRACKING hands
	 */
	int getNumHand() const;

	/**
	 * \brief Get A.R.T. FINGERTRACKING hand data.
	 *
	 * Refers to last received frame.
	 *
	 * @param[in] id Id, range 0 ..
	 * @return       Id-th A.R.T. FINGERTRACKING hand data; NULL in case of error
	 */
	const DTrackHand* getHand( int id ) const;

	/**
	 * \brief Get number of calibrated ART-Human models.
	 *
	 * Refers to last received frame.
	 *
	 * @return Number of calibrated ART-Human models
	 */
	int getNumHuman() const;

	/**
	 * \brief Get ART-Human model data.
	 *
	 * Refers to last received frame.
	 *
	 * @param[in] id Id, range 0 ..
	 * @return       Id-th ART-Human model data; NULL in case of error
	 */
	const DTrackHuman* getHuman( int id ) const;

	/**
	 * \brief Get number of calibrated hybrid (optical-inertial) bodies.
	 *
	 * Refers to last received frame.
	 *
	 * @return Number of calibrated hybrid bodies
	 */
	int getNumInertial() const;

	/**
	 * \brief Get hybrid (optical-inertial) data.
	 *
	 * Refers to last received frame.
	 *
	 * @param[in] id Id, range 0 ..
	 * @return       Id-th inertial body data; NULL in case of error
	 */
	const DTrackInertial* getInertial( int id ) const;

	/**
	 * \brief Get number of tracked single markers.
	 *
	 * Refers to last received frame.
	 *
	 * @return Number of tracked single markers
	 */
	int getNumMarker() const;

	/**
	 * \brief Get single marker data.
	 *
	 * Refers to last received frame.
	 *
	 * @param[in] index Index, range 0 ..
	 * @return          I-th single marker data; NULL in case of error
	 */
	const DTrackMarker* getMarker( int index ) const;

	/**
	 * \brief Returns if system status data is available.
	 *
	 * Refers to last received frame.
	 *
	 * @return System status data is available
	 */
	bool isStatusAvailable() const;

	/**
	 * \brief Get system status data.
	 *
	 * Refers to last received frame.
	 *
	 * @return System status data; NULL in case of error
	 */
	const DTrackStatus* getStatus() const;


private:

	/**
	 * \brief Parses a single line of frame counter data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of 'fr' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_fr( char **line );

	/**
	 * \brief Parses a single line of timestamp data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of 'ts' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_ts( char **line );

	/**
	 * \brief Parses a single line of additional information about number of calibrated bodies in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '6dcal' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_6dcal( char **line );

	/**
	 * \brief Parses a single line of standard body data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '6d' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_6d( char **line );

	/**
	 * \brief Parses a single line of 6d covariance data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '6dcov' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_6dcov( char **line );

	/**
	 * \brief Parses a single line of Flystick data (older format) data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '6df' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_6df( char **line );

	/**
	 * \brief Parses a single line of Flystick data (newer format) data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '6df2' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_6df2( char **line );

	/**
	 * \brief Parses a single line of Measurement Tool data (older format) in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '6dmt' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_6dmt( char **line );

	/**
	 * \brief Parses a single line of Measurement Tool data (newer format) data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '6dmt2' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_6dmt2( char **line );

	/**
	 * \brief Parses a single line of Measurement Tool reference data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '6dmtr' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_6dmtr( char **line );

	/**
	 * \brief Parses a single line of additional information about number of calibrated Fingertracking hands in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of 'glcal' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_glcal( char **line );

	/**
	 * \brief Parses a single line of A.R.T. Fingertracking hand data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of 'gl' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_gl( char **line );

	/**
	 * \brief Parses a single line of ART-Human model data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '6dj' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_6dj( char **line );

	/**
	 * \brief Parses a single line of hybrid (optical-inertial) body data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '6di' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_6di( char **line );

	/**
	 * \brief Parses a single line of single marker data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of '3d' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_3d( char **line );

	/**
	 * \brief Parses a single line of system status data in one tracking data packet.
	 *
	 * Updates internal data structures.
	 *
	 * @param[in,out] line Line of 'st' data in one tracking data packet
	 * @return             Parsing succeeded?
	 */
	bool parseLine_st( char** line );

private:

	unsigned int act_framecounter;                    //!< Frame counter
	double act_timestamp;                             //!< Timestamp (-1, if information not available)
	int act_num_body;                                 //!< Number of calibrated standard bodies (as far as known)
	std::vector< DTrackBody > act_body;               //!< Array containing standard body data
	int act_num_flystick;                             //!< Number of calibrated Flysticks
	std::vector< DTrackFlyStick > act_flystick;       //!< Array containing Flystick data
	int act_num_meatool;                              //!< Number of calibrated Measurement Tools
	std::vector< DTrackMeaTool > act_meatool;         //!< Array containing Measurement Tool data
	int act_num_mearef;                               //!< Number of calibrated Measurement Tool references
	std::vector< DTrackMeaRef > act_mearef;           //!< Array containing Measurement Tool reference data
	int act_num_hand;                                 //!< Number of calibrated A.R.T. FINGERTRACKING hands (as far as known)
	std::vector< DTrackHand > act_hand;               //!< Array containing A.R.T. FINGERTRACKING hand data
	int act_num_human;                                //!< Number of calibrated ART-Human models
	std::vector< DTrackHuman > act_human;             //!< Array containing ART-Human model data
	int act_num_inertial;                             //!< Number of calibrated hybrid (optical-inertial) bodies
	std::vector< DTrackInertial > act_inertial;       //!< Array containing hybrid (optical-inertial) body data
	int act_num_marker;                               //!< Number of tracked single markers
	std::vector< DTrackMarker > act_marker;           //!< Array containing single marker data
	bool act_is_status_available;                     //!< System status data is available
	DTrackStatus act_status;                          //!< System status data

	int loc_num_bodycal;    //!< internal use, local number of calibrated bodies
	int loc_num_handcal;    //!< internal use, local number of hands
	int loc_num_flystick1;  //!< internal use, local number of old flysticks
	int loc_num_meatool1;   //!< internal use, local number of old measurementtools
};


#endif  // _ART_DTRACKSDK_PARSER_HPP_

