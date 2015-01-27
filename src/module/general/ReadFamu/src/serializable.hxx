/** @file serializable.cxx
 * a serializable object.
 * FAMU Copyright (C) 1998-2006 Institute for Theory of Electrical Engineering
 * @author W. Hafla
 */

// #include "serializable.hxx" // a serializable object.

#ifndef __serializable_hxx__
#define __serializable_hxx__

#include "objectoutputstream.hxx"   // a stream that can be used for serialization.


/**
 * a serializable object.
 */
class Serializable
{
public:

    Serializable() {};
    virtual ~Serializable() {};
    
    /**
     * writes the object to an archive.
     */
    virtual void writeObject(ObjectOutputStream* archive) const = 0;
};


#endif // __serializable_hxx__





























#if 0


// ----------------------------------------------------------------------------------------------------------------
//                                  CDataStreamRecorder
// ----------------------------------------------------------------------------------------------------------------

/**
 * constructor.
 * @param filename filename of the dump file to write to/read from
 * @param modeWrite @c true if data will be written dump, @c false if data will be read
 * @param maxDumpSize maximum size of the dump file
 * @param *messages messages management
 */
CDumpRecorder::CDumpRecorder(std::string  filename, 
                             bool         modeWrite, 
                             UINT         maxDumpSize, 
                             CMessages   *messages)
: m_messages(messages),
  m_modeWrite(modeWrite),
  m_maxDumpSize(maxDumpSize),
  m_dumpIndex(0),
  m_filename(filename)
{   
    if (m_modeWrite)
    {
        m_outFile.open(filename.c_str(), std::ofstream::binary);
        if (!m_inFile)
            messages->output(Error_IO, 10, filename);
    }
    else
    {
        m_inFile.open(filename.c_str(), std::ifstream::binary);
        if (!m_outFile)
            messages->output(Error_IO, 4, filename);    // "can't open "..." for output"
    }
}//CDumpRecorder::CDumpRecorder


CDumpRecorder::~CDumpRecorder()
{
    if (m_modeWrite)
        m_outFile.close();
    else
        m_inFile.close();
}


///< writes an arbitrary number of characters to the dump file
void CDumpRecorder::putCharArr(CHAR *arr, UINT arrSize)
{
    if (!m_modeWrite)
        m_messages->output(Error_Internal, 45);             // illegal function call
    if (sizeof(CHAR)*arrSize > m_maxDumpSize)               // maximum file size reached. stop recording.
        return;
    m_outFile.write( arr, sizeof(CHAR)*arrSize);
    if (!m_outFile)
        m_messages->output(Error_IO, 24, m_filename);       // "error while trying to write to ..."
    m_dumpIndex += sizeof(CHAR)*arrSize;
}


///< writes a @c CHAR to the dump file
void CDumpRecorder::putCHAR(CHAR c)
{
    putCharArr(&c, 1);
}


///< writes an @c UCHAR to the dump file
void CDumpRecorder::putUCHAR(UCHAR u)
{
    putCharArr((CHAR*)&u, sizeof(UCHAR));
}


///< writes an @c INT to the dump file
void CDumpRecorder::putINT(INT i)
{
    putCharArr((CHAR*)&i, sizeof(INT));
}


///< writes an @c UINT to the dump file
void CDumpRecorder::putUINT(UINT u)
{
     putCharArr((CHAR*)&u, sizeof(UINT));
}


///< writes a @c double to the dump file
void CDumpRecorder::putDouble(double d)
{
    putCharArr((CHAR*)&d, sizeof(double));
}

//--------------------------------------------------------------------------

///< reads an arbitrary number of characters from the dump file
void CDumpRecorder::getCharArr(CHAR *arr, UINT arrSize)
{
    if (m_modeWrite)
        m_messages->output(Error_Internal, 45);         // illegal function call
    m_inFile.read( arr, sizeof(CHAR)*arrSize);
    if (!m_inFile)
        m_messages->output(Error_IO, 23, m_filename);   // error while trying to read from ...
    m_dumpIndex += sizeof(CHAR)*arrSize;
}


///< writes a @c CHAR to the dump file    
CHAR CDumpRecorder::getCHAR()
{
    CHAR c;
    getCharArr(&c, sizeof(CHAR));
    return c;
}


///< writes an @c UCHAR to the dump file
UCHAR CDumpRecorder::getUCHAR()
{
    UCHAR u;
    getCharArr((CHAR*)&u, sizeof(UCHAR));
    return u;
}


///< writes an @c INT to the dump file
INT CDumpRecorder::getINT()
{
    INT i;
    getCharArr((CHAR*)&i, sizeof(INT));
    return i;
}


///< writes an @c UINT to the dump file
UINT CDumpRecorder::getUINT()
{
    UINT u;
    getCharArr((CHAR*)  &u, sizeof(UINT));
    return u;
}


///< writes a @c double to the dump file
double CDumpRecorder::getDouble()
{
    double d;
    getCharArr((CHAR*)&d, sizeof(double));
    return d;
}

//--------------------------------------------------------------------------

///< compares the @c CHAR value @c c with the correspondant of the dump file
void CDumpRecorder::checkCHAR(CHAR c)
{
    UINT oldDumpIndex = m_dumpIndex;
    UCHAR dumpVal = getCHAR();
    if (dumpVal != c)
        m_messages->output(Error_Internal, 72, (double)dumpVal, c, oldDumpIndex);   // "data differs from stored dump."
}


///< compares the @c UCHAR value @c u with the correspondant of the dump file
void CDumpRecorder::checkUCHAR(UCHAR u)
{
    UINT oldDumpIndex = m_dumpIndex;
    UCHAR dumpVal = getUCHAR();
    if (dumpVal != u)
        m_messages->output(Error_Internal, 72, (double)dumpVal, u, oldDumpIndex);   // "data differs from stored dump."
}


///< compares the @c INT value @c i with the correspondant of the dump file
void CDumpRecorder::checkINT(INT i)
{
    UINT oldDumpIndex = m_dumpIndex;
    INT dumpVal = getINT();
    if (dumpVal != i)
        m_messages->output(Error_Internal, 72, (double)dumpVal, i, oldDumpIndex);   // "data differs from stored dump."
}


///< compares the @c UINT value @c u with the correspondant of the dump file
void CDumpRecorder::checkUINT(UINT u)
{
    UINT oldDumpIndex = m_dumpIndex;
    UINT dumpVal = getUINT();
    if (dumpVal != u)
        m_messages->output(Error_Internal, 72, (double)dumpVal, u, oldDumpIndex);   // "data differs from stored dump."
}


/**
 * compares the @c double value @c d with the correspondant of the dump file.
 * @param d value to check
 * @param tolerance tolerance in percent between @c d and the dump file value. deviations between @c tolerance
 *        are considered as okay/acceptable (no error is produced)
 * @param lowerLimit lower limit of the norm of @c d. if <CODE> d < lowerLimit </CODE> no check is performed.
 */
void CDumpRecorder::checkDouble(double d, double tolerance, double lowerLimit)
{
    UINT oldDumpIndex = m_dumpIndex;
    double dumpVal = getDouble();                                                   // ALWAYS input value, even if we don't check it!!!
    if (lowerLimit<1e-15 || fabs(lowerLimit)<0 )                                    // lowerLimit has to be 
        m_messages->output(Error_Internal, 45);                                     // "illegal function call"
    if (fabs(d) <= lowerLimit)
        return;
    if (fabs(dumpVal/d-1)*100 >= tolerance)
        m_messages->output(Error_Internal, 72, dumpVal, d, oldDumpIndex);           // "data differs from stored dump."
}
#endif
