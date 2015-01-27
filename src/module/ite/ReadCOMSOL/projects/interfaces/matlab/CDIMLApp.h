/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Computergenerierte IDispatch-Wrapperklassen, die mit dem Assistenten zum Hinzufügen von Klassen aus der Typbibliothek erstellt wurden

#import "C:\\Program Files\\MATLAB\\R2011b\\bin\\win64\\mlapp.tlb" no_namespace
// CDIMLApp Wrapperklasse

class CDIMLApp : public COleDispatchDriver
{
public:
    CDIMLApp()
    {
    } // Ruft den COleDispatchDriver-Standardkonstruktor auf
    CDIMLApp(LPDISPATCH pDispatch)
        : COleDispatchDriver(pDispatch)
    {
    }
    CDIMLApp(const CDIMLApp &dispatchSrc)
        : COleDispatchDriver(dispatchSrc)
    {
    }

    // Attribute
public:
    // Vorgänge
public:
    // DIMLApp Methoden
public:
    void GetFullMatrix(LPCTSTR Name, LPCTSTR Workspace, SAFEARRAY **pr, SAFEARRAY **pi)
    {
        static BYTE parms[] = VTS_BSTR VTS_BSTR VTS_UNKNOWN VTS_UNKNOWN;
        InvokeHelper(0x60010000, DISPATCH_METHOD, VT_EMPTY, NULL, parms, Name, Workspace, pr, pi);
    }
    void PutFullMatrix(LPCTSTR Name, LPCTSTR Workspace, SAFEARRAY *pr, SAFEARRAY *pi)
    {
        ///static BYTE parms[] = VTS_BSTR VTS_BSTR VTS_NONE VTS_NONE ;
        static BYTE parms[] = VTS_BSTR VTS_BSTR VTS_UNKNOWN VTS_UNKNOWN;
        InvokeHelper(0x60010001, DISPATCH_METHOD, VT_EMPTY, NULL, parms, Name, Workspace, pr, pi);
    }
    CString Execute(LPCTSTR Name)
    {
        CString result;
        static BYTE parms[] = VTS_BSTR;
        InvokeHelper(0x60010002, DISPATCH_METHOD, VT_BSTR, (void *)&result, parms, Name);
        return result;
    }
    void MinimizeCommandWindow()
    {
        InvokeHelper(0x60010003, DISPATCH_METHOD, VT_EMPTY, NULL, NULL);
    }
    void MaximizeCommandWindow()
    {
        InvokeHelper(0x60010004, DISPATCH_METHOD, VT_EMPTY, NULL, NULL);
    }
    void Quit()
    {
        InvokeHelper(0x60010005, DISPATCH_METHOD, VT_EMPTY, NULL, NULL);
    }
    CString GetCharArray(LPCTSTR Name, LPCTSTR Workspace)
    {
        CString result;
        static BYTE parms[] = VTS_BSTR VTS_BSTR;
        InvokeHelper(0x60010006, DISPATCH_METHOD, VT_BSTR, (void *)&result, parms, Name, Workspace);
        return result;
    }
    void PutCharArray(LPCTSTR Name, LPCTSTR Workspace, LPCTSTR charArray)
    {
        static BYTE parms[] = VTS_BSTR VTS_BSTR VTS_BSTR;
        InvokeHelper(0x60010007, DISPATCH_METHOD, VT_EMPTY, NULL, parms, Name, Workspace, charArray);
    }
    long get_Visible()
    {
        long result;
        InvokeHelper(0x60010008, DISPATCH_PROPERTYGET, VT_I4, (void *)&result, NULL);
        return result;
    }
    void put_Visible(long newValue)
    {
        static BYTE parms[] = VTS_I4;
        InvokeHelper(0x60010008, DISPATCH_PROPERTYPUT, VT_EMPTY, NULL, parms, newValue);
    }
    void GetWorkspaceData(LPCTSTR Name, LPCTSTR Workspace, VARIANT *pdata)
    {
        static BYTE parms[] = VTS_BSTR VTS_BSTR VTS_PVARIANT;
        InvokeHelper(0x6001000a, DISPATCH_METHOD, VT_EMPTY, NULL, parms, Name, Workspace, pdata);
    }
    void PutWorkspaceData(LPCTSTR Name, LPCTSTR Workspace, VARIANT &data)
    {
        static BYTE parms[] = VTS_BSTR VTS_BSTR VTS_VARIANT;
        InvokeHelper(0x6001000b, DISPATCH_METHOD, VT_EMPTY, NULL, parms, Name, Workspace, &data);
    }
    void Feval(LPCTSTR bstrName, long nargout, VARIANT *pvarArgOut, VARIANT &arg1, VARIANT &arg2, VARIANT &arg3, VARIANT &arg4, VARIANT &arg5, VARIANT &arg6, VARIANT &arg7, VARIANT &arg8, VARIANT &arg9, VARIANT &arg10, VARIANT &arg11, VARIANT &arg12, VARIANT &arg13, VARIANT &arg14, VARIANT &arg15, VARIANT &arg16, VARIANT &arg17, VARIANT &arg18, VARIANT &arg19, VARIANT &arg20, VARIANT &arg21, VARIANT &arg22, VARIANT &arg23, VARIANT &arg24, VARIANT &arg25, VARIANT &arg26, VARIANT &arg27, VARIANT &arg28, VARIANT &arg29, VARIANT &arg30, VARIANT &arg31, VARIANT &arg32)
    {
        static BYTE parms[] = VTS_BSTR VTS_I4 VTS_PVARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT VTS_VARIANT;
        InvokeHelper(0x6001000c, DISPATCH_METHOD, VT_EMPTY, NULL, parms, bstrName, nargout, pvarArgOut, &arg1, &arg2, &arg3, &arg4, &arg5, &arg6, &arg7, &arg8, &arg9, &arg10, &arg11, &arg12, &arg13, &arg14, &arg15, &arg16, &arg17, &arg18, &arg19, &arg20, &arg21, &arg22, &arg23, &arg24, &arg25, &arg26, &arg27, &arg28, &arg29, &arg30, &arg31, &arg32);
    }
    VARIANT GetVariable(LPCTSTR Name, LPCTSTR Workspace)
    {
        VARIANT result;
        static BYTE parms[] = VTS_BSTR VTS_BSTR;
        InvokeHelper(0x6001000d, DISPATCH_METHOD, VT_VARIANT, (void *)&result, parms, Name, Workspace);
        return result;
    }
    void XLEval(LPCTSTR bstrName, long nargout, VARIANT *pvarArgOut, long nargin, VARIANT &varArgIn)
    {
        static BYTE parms[] = VTS_BSTR VTS_I4 VTS_PVARIANT VTS_I4 VTS_VARIANT;
        InvokeHelper(0x6001000e, DISPATCH_METHOD, VT_EMPTY, NULL, parms, bstrName, nargout, pvarArgOut, nargin, &varArgIn);
    }

    // DIMLApp Eigenschaften
public:
};
