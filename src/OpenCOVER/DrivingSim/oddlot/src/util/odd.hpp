/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************
** ODD: OpenDRIVE Designer
**   Frank Naegele (c) 2010
**   <mail@f-naegele.de>
**   04.03.2010
**
**************************************************************************/

#ifndef ODD_HPP
#define ODD_HPP

class MainWindow;

class ProjectWidget;
class ProjectData;
class ChangeManager;

class ColorPalette;

#define NUMERICAL_ZERO 1.0e-15

#define NUMERICAL_ZERO3 1.0e-3
#define NUMERICAL_ZERO4 1.0e-4
#define NUMERICAL_ZERO5 1.0e-5
#define NUMERICAL_ZERO6 1.0e-6
#define NUMERICAL_ZERO7 1.0e-7
#define NUMERICAL_ZERO8 1.0e-8

/*! \brief This class contains miscellaneous identifiers und global functions.
*/
class ODD
{

    //################//
    // STATIC         //
    //################//

public:
    static ODD *instance();

    static void init(MainWindow *mainWindow);
    static void kill();

    static ProjectWidget *project(); // can't be const since it's static
    static MainWindow *mainWindow()
    {
        return mainWindow_;
    }

    static ProjectWidget *getProjectWidget();
    static ProjectData *getProjectData();
    static ChangeManager *getChangeManager();

    //################//
    // ENUMS          //
    //################//

public:
    /*! \brief Ids of the tool boxes.
	*
	* This enum defines the Id of each tool box. The Id will
	* be used by the mainToolBoxChanged(ToolBoxId) signal.
	*
	* \note Since the QToolBox has no natural Id system, the order
	* of these entries MUST BE EXACTLY THE SAME as the order of the
	* creation of the tab entries.
	*/
    enum EditorId
    {
        ERL, /*!< RoadLinkEditor */
        ERT, /*!< RoadTypeEditor */
        ETE, /*!< TrackEditor */
        ELN, /*!< LaneEditor */
        EEL, /*!< ElevationEditor */
        ESE, /*!< SuperelevationEditor */
        ECF, /*!< CrossfallEditor */
        EJE, /*!< JunctionEditor */
        ESG, /*!< SignalEditor */
        ENO_EDITOR /*!< No Editor */
    };

    /*! \brief Ids of the individual tools.
	*
	* This enum defines the Id of each tool. The Id will
	* be used by the mainToolClicked(ToolId) signal.
	*/
    enum ToolId
    {
        // Road Link Editor
        TRL_SELECT,
        TRL_LINK,
        TRL_ROADLINK,
        TRL_UNLINK,

        // Road Type Editor
        TRT_SELECT,
        TRT_ADD,
        TRT_DEL,
        TRT_MOVE,

        // Track Editor
        TTE_SELECT,
        TTE_MOVE,
        TTE_MOVE_ROTATE,
        TTE_ADD,
        TTE_ADD_LINE,
        TTE_ADD_CURVE,
        TTE_DELETE,
        TTE_TRACK_SPLIT,
        TTE_ROAD_NEW,
        TTE_ROAD_MOVE_ROTATE,
        TTE_ROAD_DELETE,
        TTE_ROAD_SPLIT,
        TTE_ROAD_MERGE,
        TTE_ROAD_SNAP,
        TTE_TRACK_ROAD_SPLIT,
        TTE_ROADSYSTEM_ADD,
        TTE_TILE_NEW,
        TTE_TILE_MOVE,
        TTE_TILE_DELETE,

        // Elevation Editor
        TEL_SELECT,
        TEL_ADD,
        TEL_DEL,
        TEL_MOVE,
        TEL_SMOOTH,

        // Superelevation Editor
        TSE_SELECT,
        TSE_ADD,
        TSE_DEL,
        TSE_MOVE,

        // Crossfall Editor
        TCF_SELECT,
        TCF_ADD,
        TCF_DEL,
        TCF_MOVE,

        // Crossfall Editor
        TLN_SELECT,

        // Lane Editor
        TLE_SELECT,
        TLE_ADD,
        TLE_DEL,
        TLE_ADD_WIDTH,
        TLE_MOVE,

        // Junction Editor
        TJE_SELECT,
        TJE_MOVE,
        TJE_SPLIT,
        TJE_CREATE_LANE,
        TJE_CREATE_ROAD,
        TJE_CREATE_JUNCTION,
        TJE_SELECT_JUNCTION,
        TJE_ADD_TO_JUNCTION,
        TJE_REMOVE_FROM_JUNCTION,
        TJE_LINK_ROADS,
        TJE_UNLINK_ROADS,
        TJE_CIRCLE,
        TJE_THRESHOLD,

        // Signal Editor
        TSG_SELECT,
        TSG_OBJECT,
        TSG_BRIDGE,
        TSG_SIGNAL,
        TSG_CONTROLLER,
        TSG_ADD_CONTROL_ENTRY,
        TSG_REMOVE_CONTROL_ENTRY,
        TSG_DEL,

        // No Tool
        TNO_TOOL
    };

    //################//
    // FUNCTIONS      //
    //################//

public:
    // Colors //
    //
    ColorPalette *colors() const
    {
        return colorPalette_;
    }

private:
    // Singleton //
    //
    ODD(); /* not allowed */
    ODD(const ODD &); /* not allowed */
    ODD &operator=(const ODD &); /* not allowed */

    //################//
    // PROPERTIES     //
    //################//

private:
    // Singleton //
    //
    static ODD *instance_;

    // MainWindow //
    //
    static MainWindow *mainWindow_;

    // Colors //
    //
    ColorPalette *colorPalette_;
};

#endif // ODD_HPP
