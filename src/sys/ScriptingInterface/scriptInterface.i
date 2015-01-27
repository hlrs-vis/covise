//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// MODULE        scriptInterface.i
//
// Description: SWIG interface definition
//
// Initial version: 10.03.2003 (rm@visenso.de)
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2003 by VirCinity IT Consulting
// (C) 2005 by Visual Engieering Solutions GmbH, Stuttgart
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
%module covise
%{

#include "coMsgStruct.h"
#include "scriptInterface.h"

%}


#if PY_VERSION_HEX >= 0x03000000
#define PyString_Check(name) PyBytes_Check(name)
#define PyString_FromString(x) PyUnicode_FromString(x)
#define PyString_Format(fmt, args)  PyUnicode_Format(fmt, args)
#define PyString_AsString(str) PyBytes_AsString(str)
#endif

// This tells SWIG to treat char ** as a special case
#if defined(SWIGPYTHON) 
%typemap(in) char ** {
    /* Check if is a list */
    if (PyList_Check($input)) {
	int size = PyList_Size($input);
	int i = 0;
	$1 = (char **) malloc((size+1)*sizeof(char *));
		//DebugBreak();
	for (i = 0; i < size; i++) {
	    PyObject *str = PyList_GetItem($input,i);
		$1[i] = SWIG_Python_str_AsChar(str);

	    //if (PyString_Check(o))
		//$1[i] = PyString_AsString(PyList_GetItem($input,i));
	    //else {
		//PyErr_SetString(PyExc_TypeError,"list must contain strings or bytes");
		//free($1);
		//return NULL;
	    //}
	}
	$1[i] = 0;
    } else {
	PyErr_SetString(PyExc_TypeError,"not a list");
	return NULL;
    }
}
#endif

// This cleans up the char ** array we malloc'd before the function call
#if defined(SWIGPYTHON) 
%typemap(freearg) char ** {
    //free((char *) $1);
	
    SWIG_Python_str_DelForPy3($1);
}
#endif

// This allows a C function to return a char ** as a Python list
#if defined(SWIGPYTHON) 
%typemap(out) char ** {
  int len, i;
  len = 0;
  if( $1 ) {
    while ($1[len] != NULL) {
      len++;
    }
  }
  $result = PyList_New(len);
  for (i = 0; i < len; i++) {
    PyList_SetItem($result,i,PyString_FromString($1[i]));
  }
}
#endif


%include coMsgStruct.h
%extend CoMsg {
    CoMsg(int t, char *d) {
	CoMsg *m;
	m = (CoMsg *) malloc( sizeof(CoMsg) );
	m->type = t;
	m->data = d;
	return m;
    }
    ~CoMsg() {
	free(self);
    }
    void show() {
	printf("CoMsg: \n type: %d \n data: %s\n",self->type,self->data); 
    }
    
    int getType() {
	return self->type;
    }

};

extern int run_xuif(int argc, char **argv);
extern int openMap(const char * fileName);
extern int runMap();
extern int clean();
extern int quit();
extern int sendCtrlMsg(char* msg);
extern int sendRendMsg(char* msg);
extern int sendErrorMsg(char* msg);
extern int sendInfoMsg(char* msg);
extern CoMsg getSingleMsg();

extern char *getCoConfigEntry(const char *entry);
extern char *getCoConfigEntry(const char *entry, const char *variable);
extern char **getCoConfigSubEntries(const char * entry);
extern bool coConfigIsOn(const char *entry);
extern bool coConfigIsOn(const char *entry, const bool &def);
extern bool coConfigIsOn(const char *entry, const char * variable);
extern bool coConfigIsOn(const char *entry, const char * variable, const bool &def);
