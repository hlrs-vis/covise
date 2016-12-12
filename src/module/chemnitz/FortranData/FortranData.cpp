/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "FortranData.h"

#ifndef _MSC_VER
#define sscanf_s sscanf
#define sprintf_s snprintf
#define fprintf_s fprintf
#define strtok_s(a, b, c) strtok(a, b)
#define strcpy_s(a, b, c) strncpy(a, c, b)
#define strncpy_s(a, b, c, d) strncpy(a, c, min(d, b))
#define stricmp strcasecmp
#define _Myptr base()
#endif

FILE *FortranData::uffFile = NULL;

FortranData::FortranData(void)
{
}

FortranData::~FortranData(void)
{
}

void FortranData::setFile(FILE *file)
{
    uffFile = file;
}

/*
read data format in fortran style (eg. 3I10, are 3 integer fields with a lenght of 10)

Input:	inputString			a line with the numbers, or text, or whatever
format				the expected line format eg. 3I10,80A1... it is separated by commata
...					pointer to the variables to be filled with data

Output:	a number in which the bit at position i represents the assigment of variable i
eg. 
If you read a line with 3 Integers with a length of 10, you would write
	int a,b,c;
	int assigned = 0;
	assigned = FortranData::ReadFortranDataFormat("3I10", &a; &b, &c);

	If everything went ok, the assigned variable should be at value 

if everything went ok, the inputString should be empty, when leaving the function
if one variable could not be assiged, its set to zero

NOTES: 

1. Pointer as 1P3E13.5 are also not processed. Use 3E13.5 instead. If you have pointer in your structures you can assign then
as followed: ReadFortranDataFormat( "3E13.5", &ptr[0], &ptr[1], &ptr[2]);

2. If a real datatype (eg. 1D25.17) is used, then you can choose either a precision (after the point) less or equal to 7 for float values, 
or greater than 7 for double values.

*/
unsigned int FortranData::ReadFortranDataFormat(const char *format, ...)
{
    if (!uffFile)
        return 0;

    va_list argList;
    void *ptr; //pointer to variable
    int i = 0; //pointer in the format string
    int j = 0; //pointer in readString
    char c = 0;
    int iStringPtr = 0; //absolute pointer in inputString
    //definition of one format set, for example if the format is 3I10 then
    int repeats = 0; //repeats is 3
    int fieldLength = 0; //fieldLength is 10
    char dataType = 0; //dataType is I
    int precision = 0; //for floating point

    char *readString; //bufferstring
    bool inputStringEndReached = false; //only, true if the end of the inputString is reached, so that remaining variables can be set to 0

    unsigned int assignedVariables = 0;
    char inputString[256]; //this is one line in the file
    unsigned int inputStringLength = 0;

    while ((c = fgetc(uffFile)) != '\n' && inputStringLength < 256)
        inputString[inputStringLength++] = c;
    if (inputStringLength > 0 && inputString[inputStringLength - 1] == '\r')
        inputStringLength--;

    inputString[inputStringLength++] = '\n';

    readString = new char[inputStringLength];
    strcpy_s(readString, inputStringLength * sizeof(char), "");

    va_start(argList, format);

    for (i = 0; i < strlen(format) - 1; i++) //go through format string
    {
        sscanf(format + i, "%d %c %d.%d", &repeats, &dataType, &fieldLength, &precision);
        memset(readString, 0, inputStringLength * sizeof(char));

        switch (dataType)
        {
        case 'I': //Integers
        {
            for (; repeats != 0; repeats--) //go through inputString
            {
                if (inputStringEndReached)
                {
                    ptr = va_arg(argList, int *);
                    //						*(int*)ptr = 0;
                    assignedVariables <<= 1;
                }
                else
                {
                    for (j = 0; j < fieldLength; j++)
                    {
                        if (iStringPtr + j >= inputStringLength)
                        {
                            inputStringEndReached = true; //end reached, before whole field could be read
                            break;
                        }
                        readString[j] = inputString[j + iStringPtr];
                    }

                    if (inputStringEndReached)
                    {
                        ptr = va_arg(argList, int *);
                        //							*(int*)ptr = 0;
                        assignedVariables <<= 1;
                        continue;
                    }

                    readString[j] = 0;
                    ptr = va_arg(argList, int *);
                    *(int *)ptr = atoi(readString);

                    iStringPtr += j;
                    assignedVariables |= 1;
                    assignedVariables <<= 1;
                }
            }
            break;
        }
        case 'A': //Strings
        {
            int t = repeats; //swap variables for fortran format
            repeats = fieldLength;
            fieldLength = t;

            for (; repeats != 0; repeats--)
            {
                if (inputStringEndReached)
                {
                    ptr = va_arg(argList, char *);
                    //						strcpy_s((char*)ptr, fieldLength, readString);
                    assignedVariables <<= 1;
                }
                else
                {
                    for (j = 0; j < fieldLength; j++)
                    {
                        if (iStringPtr + j >= inputStringLength)
                        {
                            break;
                        }
                        readString[j] = inputString[j + iStringPtr];
                    }

                    readString[j - 1] = 0;

                    ptr = va_arg(argList, char *);
                    strcpy_s((char *)ptr, fieldLength, readString); //skip spaces at the beginning of readString

                    iStringPtr += j;
                    assignedVariables |= 1; //variable could be assigned, so set bit i to 1
                    assignedVariables <<= 1;
                }
            }
            break;
        }
        case 'X': //Spaces
        {
            for (; repeats != 0; repeats--)
            {
                if (!inputStringEndReached)
                {
                    iStringPtr++;
                }
            }

            break;
        }

        case 'D': //Doubles
        case 'E':
        {
            for (; repeats != 0; repeats--)
            {
                if (inputStringEndReached)
                {
                    if (precision <= 7)
                    {
                        ptr = va_arg(argList, float *);
                        //							*(float*)ptr = 0;
                        continue;
                    }
                    else
                    {
                        ptr = va_arg(argList, double *);
                        //							*(double*)ptr = 0;
                        continue;
                    }

                    assignedVariables <<= 1; //variable i could not be assigned so just shift left
                }
                else
                {
                    for (j = 0; j < fieldLength; j++)
                    {
                        if (iStringPtr + j >= inputStringLength)
                        {
                            inputStringEndReached = true;
                            break;
                        }

                        readString[j] = inputString[j + iStringPtr];
                    }

                    if (inputStringEndReached)
                    {
                        if (precision <= 7)
                        {
                            ptr = va_arg(argList, float *);
                            //								*(float*)ptr = 0;
                            continue;
                        }
                        else
                        {
                            ptr = va_arg(argList, double *);
                            //								*(double*)ptr = 0;
                            continue;
                        }
                        assignedVariables <<= 1;
                    }

                    readString[j] = 0;

                    if (precision <= 7)
                    {
                        ptr = va_arg(argList, float *);
                        *(float *)ptr = atof(readString);
                    }
                    else
                    {
                        ptr = va_arg(argList, double *);
                        *(double *)ptr = atof(readString);
                    }

                    iStringPtr += j;
                    assignedVariables |= 1; //variable could be assigned, so set bit i to 1
                    assignedVariables <<= 1;
                }
            }
            break;
        }
        }

        while (*(format + i) != ',' && *(format + i) != 0)
            i++;

        if (iStringPtr >= inputStringLength)
            inputStringEndReached = true;
    }

    va_end(argList);

    strcpy_s(readString, inputStringLength * sizeof(char), "");
    delete readString;

    return assignedVariables >>= 1; //undo last shift
}

void FortranData::WriteFortranDataFormat(const char *f, ...)
{
    char *token;
#ifdef WIN32
    char *nextToken = NULL;
#endif
    char seps[] = ",";
    char *format = NULL;
    unsigned int formatLength = strlen(f);

    unsigned int repeats = 0;
    unsigned int fieldLength = 0;
    unsigned int precision = 0;
    char dataType = 0;

    va_list argList;
    void *argPtr = NULL;

    char *data = NULL; //data to be written, must be a string, because of the leading spaces

#ifdef _MSC_VER
#if _MSC_VER < 1900
    _set_output_format(_TWO_DIGIT_EXPONENT);
#endif
#endif

    format = new char[formatLength + 1];
    strcpy_s(format, (formatLength + 1) * sizeof(char), f);

    token = strtok_s(format, seps, &nextToken);

    va_start(argList, f);

    while (token != NULL)
    {
        sscanf(token, "%d %c %d.%d", &repeats, &dataType, &fieldLength, &precision);

        //		printf("%s %d %c %d %d \n", token, repeats, dataType, fieldLength, precision);

        switch (dataType)
        {
        case 'A':
        {
            data = new char[fieldLength + 1];
            unsigned int strLen = 0;

            for (int i = 0; i < repeats; i++)
            {
                argPtr = va_arg(argList, char *);
                strLen = strlen((const char *)argPtr);

                for (int i = 0; i < fieldLength; i++)
                {
                    if (i < strLen)
                        data[i] = ((const char *)argPtr)[i];
                    else
                        data[i] = ' ';
                }

                data[fieldLength] = 0;

                fprintf_s(uffFile, "%s", data);
            }

            delete data;
            data = NULL;

            break;
        }
        case 'I':
        {
            char *tmp = NULL;
            int pos = 0;

            data = new char[fieldLength + 1];
            tmp = new char[fieldLength + 1];

            for (int i = 0; i < repeats; i++)
            {
                argPtr = va_arg(argList, int *);

                memset(data, ' ', fieldLength + 1);

                data[fieldLength] = 0;

                sprintf_s(tmp, (fieldLength + 1) * sizeof(char), "%i", *(int *)argPtr);

                pos = strlen(data) - strlen(tmp);

                strcpy_s(data + pos, (fieldLength + 1 - pos) * sizeof(char), tmp);

                fprintf_s(uffFile, "%s", data);
            }

            delete tmp;
            delete data;
            tmp = NULL;
            data = NULL;

            break;
        }
        case 'X':
        {
            data = new char[repeats + 1];

            memset(data, ' ', repeats);
            data[repeats] = 0;

            fprintf_s(uffFile, "%s", data);

            delete data;
            data = NULL;

            break;
        }
        case 'D':
        case 'd':
        case 'E':
        case 'e':
        {
            char *tmp = NULL;
            char formatString[8]; //sprintf format string, eg. %xx.yyf

            data = new char[fieldLength + 1];
            tmp = new char[fieldLength + 1];

            for (int i = 0; i < repeats; i++)
            {
                memset(data, ' ', fieldLength + 1);
                data[fieldLength] = 0;

                sprintf_s(formatString, 8 * sizeof(char), "%%%d.%de", fieldLength, precision);

                if (precision <= 7)
                {
                    argPtr = va_arg(argList, float *);
                    sprintf_s(tmp, (fieldLength + 1) * sizeof(char), formatString, *(float *)argPtr);
                }
                else
                {
                    argPtr = va_arg(argList, double *);
                    sprintf_s(tmp, (fieldLength + 1) * sizeof(char), formatString, *(double *)argPtr);
                }

                tmp[fieldLength - 4] = dataType;

                fprintf(uffFile, "%s", tmp);
            }

            delete tmp;
            delete data;
            tmp = NULL;
            data = NULL;

            break;
        }
        }

        token = strtok_s(NULL, seps, &nextToken);

        repeats = 0;
        fieldLength = 0;
        precision = 0;
        dataType = 0;
    }

    fprintf_s(uffFile, "\n");

    va_end(argList);
}
