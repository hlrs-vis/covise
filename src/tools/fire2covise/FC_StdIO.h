/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//----------------------------------------------------------------------//
//   $Id: FC_StdIO.h,v 1.7 2000/03/22 14:21:49 goesslet Exp $
//----------------------------------------------------------------------//
//
//   $Log: FC_StdIO.h,v $
//   Revision 1.7  2000/03/22 14:21:49  goesslet
//   direct goto header of dataset
//
//   Revision 1.6  2000/03/17 23:01:46  goesslet
//   error writing long int on platform where long
//   is only 4 bytes
//
//   Revision 1.5  2000/02/17 11:42:51  goesslet
//   added virtal to destructor
//
//   Revision 1.4  2000/02/14 16:11:57  goesslet
//   implementation of compression
//
//   Revision 1.3  1999/12/15 08:46:29  goesslet
//   removed platform dependent includes
//
//   Revision 1.2  1999/11/23 09:58:23  goesslet
//   programmName and programmVersion now available in FC_Base
//
//   Revision 1.1  1999/11/16 14:13:58  goesslet
//   initial version of FC_StdIO
//
//----------------------------------------------------------------------//

#ifndef _FC_STDIO_H_
#define _FC_STDIO_H_

#include "FC_Base.h"
#include "FC_String.h"
#include "FC_ListArray.h"

#define FC_STDIO_HEADER_LEN 720

class FC_StdIO : public FC_Base
{
    friend int operator!(FC_StdIO & file);

public:
    // mode for open file
    enum eMode
    {
        read,
        write,
        append
    };
    enum eAcess
    {
        ascii,
        bytestream,
        binary
    };
    enum eFormat
    {
        formatted,
        unformatted,
        formattedText
    };

    // error which can happen
    enum eStatus
    {
        ok,
        error,
        no_error,
        error_open,
        error_read,
        error_write,
        error_format,
        error_access,
        error_seek,
        error_data_version,
        error_data_name,
        error_data_type,
        error_conversion,
        error_close,
        error_end,
        error_data_compression,
        error_data_uncompression,
        error_byteswaptype
    };
    // datatypes which can be write
    enum eDataType
    {
        data_int,
        data_float,
        data_double,
        data_char,
        data_longint
    };

public:
    FC_StdIO();
    virtual ~FC_StdIO();

    eStatus Open(const FC_String &fileNameIn,
                 const eMode &mode,
                 const eAcess &type = FC_StdIO::binary,
                 const eFormat &fmt = FC_StdIO::unformatted);

    virtual eStatus Close();

    void PrintFileInfo(ostream &stream = cout) const;

    FC_String CurrentHeader();

    // -------------------------------------------------------------------
    // functions for reading data

    // functions for accesing data sets
    // returns highest available dataset
    int NumberOfDataSets();
    // check if data set is available
    eStatus DataSetIsAvailable(const int &dataSet);

    // got dataset in file
    eStatus GotoFirstDataSet();
    eStatus GotoDataSet(const int &dataSet);
    eStatus GotoDataSetEnd(const int &dataSet);
    // check if we are in the specified data set
    eStatus InDataSet(const int &dataSet);

    // reading data header
    eStatus ReadDataHeader(int &version,
                           FC_String &name,
                           eDataType &type,
                           int &numData);
    // goto header int specified dataset
    eStatus GotoDataHeader(const int &dataSet, const FC_String &nameIn,
                           int &version, eDataType &type, int &numData);
    // goto next header of dataset
    eStatus GotoNextHeader();

    // read value from file
    // -------------------------------------------------------------------
    eStatus Read(int &valueIn, const int &fromBuffer = 1);
    eStatus Read(float &valueIn, const int &fromBuffer = 1);
    eStatus Read(double &valueIn, const int &fromBuffer = 1);
    eStatus Read(long int &valueIn, const int &fromBuffer = 1);

    // read array from file
    eStatus Read(const int &num, FC_Array<int> &valueOut);
    eStatus Read(const int &num, FC_Array<float> &valueOut);
    eStatus Read(const int &num, FC_Array<double> &valueOut);
    eStatus Read(const int &num, FC_Array<long int> &valueOut);
    eStatus Read(const int &num, FC_String &valueOut);

    // -------------------------------------------------------------------

    // -------------------------------------------------------------------
    // functions for writing data

    // creating new dataset
    eStatus NewDataSet(const int &dataSet);
    eStatus EndDataSet(const int &dataSet);

    // write value to file
    // -------------------------------------------------------------------
    eStatus Write(const int &valueIn, const int &toBuffer = 1);
    eStatus Write(const float &valueIn, const int &toBuffer = 1);
    eStatus Write(const double &valueIn, const int &toBuffer = 1);
    eStatus Write(const long int &valueIn, const int &toBuffer = 1);

    // write array to file
    eStatus Write(const FC_Array<int> &valueIn);
    eStatus Write(const FC_Array<float> &valueIn);
    eStatus Write(const FC_Array<double> &valueIn);
    eStatus Write(const FC_Array<long int> &valueIn);
    eStatus Write(const FC_String &valueIn);

    // write data header
    eStatus WriteDataHeader(const int &version,
                            const FC_String &name,
                            const eDataType &type,
                            const int &numData);
    // -------------------------------------------------------------------
    // turn on or off display of error messages
    void DisplayErrorMessages(int onOff);

    // -------------------------------------------------------------------------------------------------
private:
    eStatus readFileInfo();
    eStatus writeFileInfo();
    eStatus readContext();
    eStatus writeContext();
    eStatus errorCheck();

    void readBytes(char *valueIn, const int &numBytes, const int &fromBuffer);
    void writeBytes(char *valueIn, const int &numBytes, const int &toBuffer);

    void setupByteSwapType();
    void swapBytes(char *buffer, const int &numBytes);

    // functions for reading and writing bytstream
private:
    // real open function of file
    void openFile(const FC_String &mode);
    // real close function of file
    void closeFile();
    int setFilePointer(const long int &pos, const int &wence);
    long int getFilePointer();

    eStatus openByteStream();
    eStatus writeHeaderByteStream(const FC_String &head1,
                                  const FC_String &head2,
                                  const FC_String &head3);

    eStatus readContextByteStream();
    eStatus writeContextByteStream();

    eStatus readDataHeaderByteStream(int &version,
                                     FC_String &name,
                                     eDataType &type,
                                     int &numData);
    eStatus readByteStream(const int &num, int *valueOut);
    eStatus readByteStream(const int &num, float *valueOut);
    eStatus readByteStream(const int &num, double *valueOut);

    eStatus writeDataHeaderByteStream(const int &version,
                                      const FC_String &name,
                                      const eDataType &type,
                                      const int &numData);
    eStatus writeByteStream(const int &num, const int *valueIn);
    eStatus writeByteStream(const int &num, const float *valueIn);
    eStatus writeByteStream(const int &num, const double *valueIn);

    // display error message if status is different to no_error
    void errorMessage();

    char char_value[8];
    FC_String currentHeader;
    eDataType currentDataType;
    int numberOfDataSets;
    long int posOfContext;

    // number of value in each line, number of written data
    // only for ascii mode used
    int numDataToRead;
    int numDataPerLine, numDataWritten, numDataRead;
    void addEndOfLine();

    FC_ListArray<long int> startPosOfDataSet, endPosOfDataSet;

    FILE *id;
    FC_String fileName;

    enum eStatus status;
    enum eMode openmode;
    enum eFormat format;
    enum eAcess access;

    int displayErrorMessages;
    int needByteSwap;

    // -------------------------------------------------------------------
    // for bytestream stuff
    // -------------------------------------------------------------------
    int recordLen;
    int recordCounter;
    int headerRecordCounter;
    int currentNumberOfRecords;
    int currentBytesPerUnit;

    char stdCharBuf[FC_STDIO_HEADER_LEN];

    int crcConvertType;
    int intConvertType;
    int floatConvertType;

    FC_ListArray<char> dataBuffer;
    void clearDataBuffer();
    void addToDataBuffer(const char *b, const int &nBytes);
    void getFromDataBuffer(char *b, const int &nBytes);
    eStatus writeDataBuffer();
    eStatus readDataBuffer();

public:
    static const int stdIntValue;
    static const float stdFloatValue;
    // -------------------------------------------------------------------

private:
    // store number of openfiles and highest fortran unit
    static int numOpenFiles;
};

int operator!(const FC_StdIO &file);
ostream &operator<<(ostream &stream, const FC_StdIO &file);
#endif
