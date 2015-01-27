/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file ReadDTF.h
 * @brief declaration of class ReadDTF
 *
 * This file contains the declaration of class ReadDTF.
 *
 * @author Alexander Martinez, HSG-IMIT VS
 * @date 17.9.2003
 * created by module_gen
 */

/** @class ::ReadDTF
 * @brief inherited from coModule
 *
 * This class is used to parse and process CFD data stored in files with a
 * special format. Those files must be saved in CFD-DTF format.
 *
 * Objects of this class are used as COVISE plugin module and are (ab)used
 * there to visualize CFD data.
 *
 * \b Use:
 * @code
 * int main(int argc, char* argv[])
 * {
 *   ReadDTF* application = new ReadDTF;
 *   application->start(argc, argv);
 *
 *   return 0;
 * }
 * @endcode
 *
 * @see main() for the current implementation in ReadDTF module.
 */

/** @fn virtual int ReadDTF::compute(const char *port)
 * @brief covise execution event
 *
 * This function is called out of the main execution loop of the module. That
 * loop is entered in \c main() by calling \c start().
 *
 * Each time Covise calls selfExec or when the user chooses to execute the
 * module then this function is entered to retrieve and process the data
 * saved into a given DTF file. If \a updateDataOnly is \c true then
 * the data display is only updated - no fresh data is loaded from DTF file
 * until it is necessary.
 */

/** @fn ReadDTF::ReadDTF()
 * @brief default constructor for Covise module.
 *
 * Does some basic initialization. Covise sets timeout for
 * constructor - do all non-basic initialisation in \a postInst().
 *
 * This constructor creates the parameters and ports used by the ReadDTF module.
 */

/** @fn virtual void ReadDTF::postInst();
 * @brief time-critical initializations
 *
 * Performs any time-critical initializations which couldn't be done in
 * constructor.
 *
 * It tells all parameters to be shown in Covise GUI (by calling \c show() ).
 */

/** @fn virtual void ReadDTF::param(const char* paramName);
 * @brief called when immediate params change
 *
 * @param paramName - name of the parameter which value has changed.
 *
 * This function is called when the data parameters of the module are changed.
 * It sets \a updateDataOnly to \c true and calls \c selfExec() to redraw the
 * scene and update the data ports. The geometry and data of the underlying
 * DTF file is only retrieved when there is no data stored in \a wrapper. That
 * means that the DTF file hasn't been read before.
 *
 * Immediate parameters are:
 * - all data parameters
 * - tsStartParam ( specifies first time step )
 * - tsEndParam ( specifies last time step )
 */

/** @fn virtual void ReadDTF::quit();
 * @brief called before module exits
 *
 * Since the destructor of ReadDTF is never called, this function is needed
 * to perform some memory cleanups.
 *
 * Basically it removes the contents of \a wrapper from memory to reduce memory
 * leaks produced by the module.
 *
 * Additionally all data ports and data parameters are deleted.
 *
 * @note This function can't reduce memory leaks produced by Covise. It only
 * clears its own occupied memory.
 */

/** @fn void ReadDTF::fillTypeList();
 * @brief fills \a dataTypes with list of supported data types
 *
 * The created list is used inside Covise by the user to specify which type of
 * data should be sent to which data port.
 *
 * Supported types are:
 * - "None"
 * - "Cell_U"
 * - "Cell_V"
 * - "Cell_W"
 * - "Cell_RHO"
 * - "Cell_P"
 * - "Cell_MU"
 * - "Cell_T"
 * - "RHO"
 * - "U"
 * - "V"
 * - "W"
 * - "P"
 * - "P_tot"
 * - "Vislam"
 * - "T"
 *
 * Currently the supported types are static. This may change later.
 */

/** @fn void ReadDTF::updateDataPort();
 * @brief prepares data for Covise output
 *
 * If wrapper isn't NULL then this function iterates through all dataParams
 * and dataPorts and sents the according output data to the selected output
 * ports.
 *
 * @note Data output depends on that data which the user selected in the Covise
 * GUI.
 */

/** @fn bool ReadDTF::createDataParams();
 * @brief creates the list of available dataParams.
 *
 * @return always true
 *
 * This function creates the available data parameters visible to the user in
 * the Covise GUI. Those data parameters are used to specify the type of data
 * which should be sent through the according data ports.
 */

/** @fn bool ReadDTF::createDataPorts();
 * @brief creates the data ports for coDoFloat
 *
 * @return always true
 *
 * This function creates the available data ports for module ReadDTF. Actual
 * output depends on values of dataParams.
 */

/** @var coOutputPort* ReadDTF::gridPort;
 * @brief output port for unstructured grids
 *
 * This port is used to deliver sets of unstructured grids (of type
 * DO_Unstructured_Grid) read from DTF to Covise for further processing.
 *
 * They can be used directly by the renderer. Each content of the set represents
 * one time step in the simulation.
 */

/** @var coFileBrowserParam* ReadDTF::fileParam;
 * @brief used to choose the DTF file to parse
 *
 * This parameter is used to select the DTF file which should be parsed and
 * which data should be processed by the ReadDTF module.
 */

/** @var DataWrapper* ReadDTF::wrapper;
 * @brief reads and prepares DTF data
 *
 * This object is used inside \a compute(const char *port) to read, process and prepare DTF
 * data read from a DTF file. That data is used later to create objects
 * of type coDoSet, which contain
 * - coDoUnstructuredGrid objects for the simulation geometry
 * - coDoFloat for the data associated with the mesh nodes
 */

/** @var vector<coOutputPort*> ReadDTF::dataPorts;
 * @brief contains available output ports for data
 *
 * To support multiple data outputs at the same time multiple data ports are
 * required. Each of that ports can provide data output for the available
 * data types read from DTF file.
 *
 * @note data may be of following types:
 * - Cell based data (wrap outgoing grid + data through module Cell2Vert)
 * - Node based data (normal processing of output grids + data)
 */

/** @var vector<coChoiceParam*> ReadDTF::dataParams;
 * @brief data parameters shown in Covise GUI
 *
 * This parameters are shown in the Covise GUI and provide an easy way to
 * select the data type which is to be sent through one of the \a dataPorts.
 */

/** @var coBooleanParam* ReadDTF::isIndexParam;
 * @brief specifies if supplied file name in \a fileParam is index file.
 *
 * ReadDTF supports two operation modes:
 * - single DTF file ( no time steps )
 * - multiple DTF files specified by index file ( time steps)
 */

/** @var coIntScalarParam* ReadDTF::tsStartParam;
 * @brief First time step in simulation.
 *
 * Used to specify the starting frame for a time step animation read from
 * multiple DTF files.
 *
 * If this parameter is smaller than 0 or bigger than number of time steps then
 * it is meaningless - \c 0 is used instead.
 *
 * This parameter is only usefull when multiple DTF files are read in.
 *
 * Default is 0.
 */

/** @var coIntScalarParam* ReadDTF::tsEndParam;
 * @brief last time step in simulation.
 *
 * Used to specify the last frame for a time step animation read from multiple
 * DTF files.
 *
 * If this parameter is bigger than the number of available time steps then
 * it is meaningless - the last available time step is used instead.
 *
 * This parameter is only usefull when multiple DTF files are read in.
 *
 * Default value is \c 100.
 */

/** @var map<int, string> ReadDTF::dataTypes;
 * @brief list of available data types for data ports.
 *
 * DTF files contain different types of data:
 * - Cell based data (1 entry for each cell)
 * - Node based data (1 entry for each entry in coordinate list)
 *
 * @see fillTypeList() for a concrete list of supported data types
 */

/** @var bool ReadDTF::updateDataOnly;
 * @brief indicates if data is to be read completely or to be displayed only
 *
 * I user changes data types in Covise GUI then this variable is set to \c true.
 * This avoids that complex DTF files are read in completely without need.
 *
 * If set to \c true then the geometry and data contained in \a wrapper is
 * sent to Covise which performs an update of its internal data store.
 *
 * default: \c false
 */

#ifndef _ReadDTF_H_
#define _ReadDTF_H_

#include "DataWrapper.h"

class ReadDTF : public coModule
{
private:
    coBooleanParam *isIndexParam;
    coFileBrowserParam *fileParam;
    coIntScalarParam *tsStartParam;
    coIntScalarParam *tsEndParam;

    vector<coChoiceParam *> dataParams;
    vector<coOutputPort *> dataPorts;

    coOutputPort *gridPort;
    DataWrapper *wrapper;

    map<int, string> dataTypes;

    bool updateDataOnly;

    void fillTypeList();
    void updateDataPort();
    bool createDataParams();
    bool createDataPorts();

public:
    ReadDTF();

    virtual int compute(const char *port);
    virtual void postInst();
    virtual void param(const char *paramName);
    virtual void quit();
};
#endif
