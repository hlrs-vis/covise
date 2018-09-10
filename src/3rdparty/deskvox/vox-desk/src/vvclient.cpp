// DeskVOX - Volume Exploration Utility for the Desktop
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
// 
// This file is part of DeskVOX.
//
// DeskVOX is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// 
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the 
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

// Compiler:
#include <iostream>

// Local:
#include "vvclient.h"

using namespace vox;
using namespace std;
#ifdef WIN32
  #ifdef HAVE_SOAP
    using namespace MSSOAPLib30;
    using namespace MSXML2;
  #endif
#endif

/** 
  Key: SQLCTClient.WebReference.EP_SQLCT
  This routine depends on the data format defined in ClientForm.cs on the
  server side. The following routines are supported by the server:
  
  GetRegion(datasetID, lod, x0, y0, x1, y1, startSlice, endSlice);
  GetDatasets(): returns an array of dataset names and ids
  GetDatasetSliceCount(id): returns the slice count of a dataset. This is how you would 
    know what slice values to pass the GetRegion method. 

  @return pointer to transferred volume data set, or NULL on error
  @param lod level of detail, corresponds to mipmap level. 
     For each slice, we generate 5 mipmap levels (1024x1024 - 64x64). 
     Then, for each level, we construct a quadtree of subregions until 
     we reach 64x64. For example, mipmap level 4 = 512 x 512. 
     It has 1 region of 512 x 512 and 4 subregions of 128 x 128 and 
     16 subregions of 64 x 64.
  @param server name of data server
  @param ns SOAP name space
*/
unsigned char* vvClient::getRegion(int lod, int x0, int y0, int x1, int y1, 
  int startSlice, int endSlice, int id, const char* server, const char* ns)
{
  cerr << "LOD:         " << lod << endl;
  cerr << "x0:          " << x0 << endl;
  cerr << "y0:          " << y0 << endl;
  cerr << "x1:          " << x1 << endl;
  cerr << "y1:          " << y1 << endl;
  cerr << "start slice: " << startSlice << endl;
  cerr << "end slice:   " << endSlice << endl;
  cerr << "ID:          " << id << endl;
  cerr << "Server:      " << server << endl;
  
//  test(); return NULL;
  return getRegionHighLevel(lod, x0, y0, x1, y1, startSlice, endSlice, id, server, ns);
//  return getRegionLowLevel(lod, x0, y0, x1, y1, startSlice, endSlice, id, server, ns);
}

/** Test program from the web for high level SOAP communication.
*/
void vvClient::test()
{
#ifdef HAVE_SOAP
  const char* DOMAIN_NAME = "google.net";
  const char* g_lpszWSDL_URL = 
      "http://services.xmethods.net/soap/urn:xmethods-DomainChecker.wsdl";

#ifdef WIN32
  HRESULT hr = CoInitialize(NULL);
  if (FAILED(hr))
  {
    cerr << "CoInitialize failed" << endl;
    return;
  }

  USES_CONVERSION;

  CComPtr<ISoapClient> spSOAPClient;
  hr = spSOAPClient.CoCreateInstance(CLSID_SoapClient30);
  if (FAILED(hr))
  {
    cerr << "spSOAPClient.CoCreateInstance failed" << endl;
    return;
  }

  cerr << "Using WSDL file: " << g_lpszWSDL_URL << endl;
  hr = spSOAPClient->MSSoapInit(_bstr_t(g_lpszWSDL_URL), L"", L"", L"");
  if (FAILED(hr))
  {
    cerr << "spSOAPClient->MSSoapInit failed" << endl;
    return;
  }

  //  Call the Web Service method
  WCHAR* pwcMethodName = L"checkDomain";
  DISPID dispidFn = 0;
  hr = spSOAPClient->GetIDsOfNames(IID_NULL, &pwcMethodName, 1, 
    LOCALE_SYSTEM_DEFAULT, &dispidFn);
  if (FAILED(hr))
  {
    cerr << "spSOAPClient->GetIDsOfNames failed" << endl;
    return;
  }

  unsigned int uArgErr;
  VARIANT varg[1];
  varg[0].vt = VT_BSTR;
  varg[0].bstrVal = _bstr_t(DOMAIN_NAME);

  DISPPARAMS params;
  params.cArgs = 1;
  params.rgvarg = varg;
  params.cNamedArgs    = 0;
  params.rgdispidNamedArgs = NULL;

  _variant_t result;

  uArgErr = (unsigned int)-1;

  EXCEPINFO excepInfo;
  memset(&excepInfo, 0, sizeof(excepInfo));
  hr  = spSOAPClient->Invoke(dispidFn, IID_NULL, LOCALE_SYSTEM_DEFAULT, 
    DISPATCH_METHOD, &params, &result, &excepInfo, &uArgErr);
  if (FAILED(hr))
  {
    cerr << "spSOAPClient->Invoke failed" << endl;
    return;
  }

  if(result.vt == VT_BSTR)
  {
    cerr << _T("Domain ") << W2A(_bstr_t(DOMAIN_NAME)) << " is " << W2A(result.bstrVal) << endl;;
  }

  CoUninitialize();
#endif
#endif
}

/** This function has been modeled after http://perfectxml.com/CPPSOAP.asp, Section 3
*/
unsigned char* vvClient::getRegionHighLevel(int lod, int x0, int y0, int x1, int y1, 
  int startSlice, int endSlice, int id, const char*, const char*)
{
  unsigned char* ptr = NULL;
#ifdef HAVE_SOAP
  const char* WSDL_URL = 
      "http://129.114.6.157/region?wsdl";
//      "Z:\\projects\\tacc\\domain-checker.wsdl";
//      "Z:\\projects\\tacc\\region-current.wsdl";
//      "http://services.xmethods.net/soap/urn:xmethods-DomainChecker.wsdl";
//    "D:\\brown\\tacc\\region.wsdl";
//    "D:\\brown\\tacc\\xmethods-DomainChecker.wsdl";

#ifdef WIN32
  HRESULT hr;
  hr = CoInitialize(NULL);
  if (FAILED(hr))
  {
    cerr << "Error: CoInitialize failed" << endl;
    return NULL;
  }

  USES_CONVERSION;

  CComPtr<ISoapClient> spSOAPClient;
  hr = spSOAPClient.CoCreateInstance(CLSID_SoapClient30);
  if (FAILED(hr))
  { 
    cerr << "Error: CoCreateInstance failed ";
    if (hr==REGDB_E_CLASSNOTREG) cerr << "(CLASSNOTREG): is MS SOAP Toolkit 3.0 installed?" << endl;
    else cerr << "hr=" << hr << endl;
    return NULL;
  }

  cerr << "Using WSDL file: " << WSDL_URL << endl;
  try
  {
//    spSOAPClient->PutConnectorProperty("AuthUser", "PRODUCTION\\webService");
    spSOAPClient->PutConnectorProperty("AuthUser", "PRODUCTION\\webservice");
    spSOAPClient->PutConnectorProperty("AuthPassword", "webservice");
    spSOAPClient->PutConnectorProperty("EndPointURL", "http://129.114.6.157/region/?wsdl");
//    hr = spSOAPClient->MSSoapInit(_bstr_t(WSDL_URL), L"", L"", L"");  // WSDL, service, port, WSML
    hr = spSOAPClient->MSSoapInit(_bstr_t(WSDL_URL), L"EP_SQLCT", L"EP_SQLCT", L"");
  }
  catch(_com_error &e)
  {
    cerr << "Exception: spSOAPClient->MSSoapInit failed. ";
    hr = e.Error();
    if (FAILED(hr))
    {
      if (hr==E_INVALIDARG) cerr << "E_INVALIDARG" << endl;
      else if (hr==OLE_E_BLANK) cerr << "OLE_E_BLANK" << endl;
      else cerr << "hr=" << hr << endl;
    }
    return NULL;
  }
  if (FAILED(hr))
  {
    cerr << "Error: spSOAPClient->MSSoapInit failed" << endl;
    return NULL;
  }

  // Set login information
  // (find more information and parameters at
  // http://www.c-sharpcorner.com/Code/2004/May/SOAPClient.asp):
//  spSOAPClient->PutConnectorProperty("AuthUser", "PRODUCTION\\webservice");
  spSOAPClient->PutConnectorProperty("AuthUser", "PRODUCTION\\webService");
  spSOAPClient->PutConnectorProperty("AuthPassword", "webservice");
//  spSOAPClient->PutConnectorProperty("WinHTTPAuthScheme", "24"); // or 1?

  // Call the Web Service method:
  WCHAR* pwcMethodName = L"GetRegion";
  DISPID dispidFn = 0;
  hr = spSOAPClient->GetIDsOfNames(IID_NULL, &pwcMethodName, 1, 
    LOCALE_SYSTEM_DEFAULT, &dispidFn);
  if (FAILED(hr))
  {
    cerr << "Error: GetIDsOfNames failed" << endl;
    return NULL;
  }

  // Set parameter set for call to 
  // GetRegion(lod, x0, y0, x1, y1, startSlice, endSlice, id):
  unsigned int uArgErr;
  VARIANT varg[8];
  varg[0].vt = VT_INT;
  varg[0].intVal = lod;
  varg[1].vt = VT_INT;
  varg[1].intVal = x0;
  varg[2].vt = VT_INT;
  varg[2].intVal = y0;
  varg[3].vt = VT_INT;
  varg[3].intVal = x1;
  varg[4].vt = VT_INT;
  varg[4].intVal = y1;
  varg[5].vt = VT_INT;
  varg[5].intVal = startSlice;
  varg[6].vt = VT_INT;
  varg[6].intVal = endSlice;
  varg[7].vt = VT_INT;
  varg[7].intVal = id;

  DISPPARAMS params;
  params.cArgs = 8;
  params.rgvarg = varg;
  params.cNamedArgs = 0;
  params.rgdispidNamedArgs = NULL;

  _variant_t result;

  uArgErr = (unsigned int)-1;

  cerr << "spSOAPClient->Invoke" << endl;
  EXCEPINFO excepInfo;
  memset(&excepInfo, 0, sizeof(excepInfo));
  hr  = spSOAPClient->Invoke(dispidFn, IID_NULL, LOCALE_SYSTEM_DEFAULT, 
    DISPATCH_METHOD, &params, &result, &excepInfo, &uArgErr);
  if (FAILED(hr))
  {
    cerr << "Error: Invoke failed" << endl;
    return NULL;
  }

  if(result.vt != VT_PTR)
  {
    cerr << "Error: Invalid return type" << endl;
    return NULL;
  }
  ptr = (unsigned char*)result.pcVal;

  CoUninitialize();
#endif
#else
  (void)lod;
  (void)x0;
  (void)y0;
  (void)x1;
  (void)y1;
  (void)startSlice;
  (void)endSlice;
  (void)id;
#endif
  
  return ptr;
}

/** This function has been inspired by http://perfectxml.com/CPPSOAP.asp, Section 4
  Another good source of information is
    http://www.codeguru.com/Cpp/COM-Tech/complus/soap/article.php/c3945/

  Namespace should be: http://sqlct.tacc.utexas.edu
*/
unsigned char* vvClient::getRegionLowLevel(int lod, int x0, int y0, int x1, int y1, 
  int slice0, int slice1, int dataset_id, const char* server, const char* ns)
{
  unsigned char* ptr = NULL;

#ifdef WIN32
#ifdef HAVE_SOAP
  char buf[32];   // for itoa
  HRESULT hr = S_OK;

  hr = CoInitialize(NULL);
  if (FAILED(hr))
  {
    cerr << "Error: CoInitialize failed" << endl;
    return NULL;
  }

  ISoapSerializerPtr Serializer;
  ISoapReaderPtr Reader;
  ISoapConnectorPtr Connector;

  cerr << "Connector.CreateInstance" << endl;
  hr = Connector.CreateInstance(__uuidof(HttpConnector30));

  cerr << "Connector->Property" << endl;
  Connector->Property[_T("EndPointURL")] = _T(server);

  cerr << "Connector->Connect" << endl;
  hr = Connector->Connect();

  cerr << "Connector->Property" << endl;
  Connector->Property[_T("SoapAction")] = _T(server);

  
  cerr << "Connector->BeginMessage" << endl;
  hr = Connector->BeginMessage();

  cerr << "Serializer.CreateInstance" << endl;
  hr = Serializer.CreateInstance(__uuidof(SoapSerializer30));

  cerr << "Serializer->Init" << endl;
  hr = Serializer->Init(_variant_t((IUnknown*)Connector->InputStream));

  cerr << "Serializer->StartEnvelope" << endl;
  hr = Serializer->StartEnvelope(_T(ns),_T("NONE"),_T(""));
  
  // Opening <body>:
  cerr << "Serializer->StartBody" << endl;
  hr = Serializer->StartBody(_T(""));
  
  // Create XML message:
  cerr << "Serializer->StartElement" << endl;
  hr = Serializer->StartElement(_T("GetRegion"), _T(""), _T("NONE"),_T(""));
  
  hr = Serializer->StartElement(_T("lod"), _T(""), _T("NONE"),_T(""));
  hr = Serializer->WriteString(itoa(lod, buf, 10));
  hr = Serializer->EndElement();
            
  hr = Serializer->StartElement(_T("x0"), _T(""), _T("NONE"),_T(""));
  hr = Serializer->WriteString(itoa(x0, buf, 10));
  hr = Serializer->EndElement();

  hr = Serializer->StartElement(_T("y0"), _T(""), _T("NONE"),_T(""));
  hr = Serializer->WriteString(itoa(y0, buf, 10));
  hr = Serializer->EndElement();

  hr = Serializer->StartElement(_T("x1"), _T(""), _T("NONE"),_T(""));
  hr = Serializer->WriteString(itoa(x1, buf, 10));
  hr = Serializer->EndElement();

  hr = Serializer->StartElement(_T("y1"), _T(""), _T("NONE"),_T(""));
  hr = Serializer->WriteString(itoa(y1, buf, 10));
  hr = Serializer->EndElement();

  hr = Serializer->StartElement(_T("slice0"), _T(""), _T("NONE"),_T(""));
  hr = Serializer->WriteString(itoa(slice0, buf, 10));
  hr = Serializer->EndElement();

  hr = Serializer->StartElement(_T("slice1"), _T(""), _T("NONE"),_T(""));
  hr = Serializer->WriteString(itoa(slice1, buf, 10));
  hr = Serializer->EndElement();

  hr = Serializer->StartElement(_T("dataset_id"), _T(""), _T("NONE"),_T(""));
  hr = Serializer->WriteString(itoa(dataset_id, buf, 10));
  hr = Serializer->EndElement();

  hr = Serializer->EndElement();

  // </body>
  cerr << "Serializer->EndBody" << endl;
  hr = Serializer->EndBody();

  cerr << "Serializer->EndEnvelope" << endl;
  hr = Serializer->EndEnvelope();
  
  cerr << "Connector->EndMessage" << endl;
  try
  {
    hr = Connector->EndMessage();    
    if (FAILED(hr))
    {
      cerr << "Error: Connector->EndMessage failed" << endl;
      return NULL;
    }
  }
  catch(...)
  {
    cerr << "Error in Connector->EndMessage" << endl;
    return NULL;
  }

  cerr << "Reader.CreateInstance" << endl;
  try
  {
    hr = Reader.CreateInstance(__uuidof(SoapReader30));
    if (FAILED(hr))
    {
      cerr << "Error: Reader.CreateInstance failed" << endl;
      return NULL;
    }
  }
  catch(...)
  {
    cerr << "Error in Reader.CreateInstance" << endl;
    return NULL;
  }

  cerr << "Reader->Load" << endl;
  try
  {
    hr = Reader->Load(_variant_t((IUnknown*)Connector->OutputStream), _T(""));
  }
  catch (_com_error e)
  {
    cerr << "_com_error in Reader->Load: " << hr << " " << FAILED(hr) << endl;
    return NULL;
  }
  catch(...)
  {
    cerr << "Error in Reader->Load" << endl;
    return NULL;
  }

  // also try: printf("Answer: %s\n", (const char *)Reader->RPCResult->text); 

  // and also MSXML2::IXMLDOMElementPtr ptr = Reader->RpcResult->get_xml((BSTR*)&bstrXml);


  cerr << "Reader->Dom" << endl;
  CComQIPtr<IXMLDOMDocument2> spResponseXMLDOM;
  spResponseXMLDOM = Reader->Dom;

  //TODO: Process the response SOAP XML and display the results
  //For now, just printing the response XML text as it is.
  
  USES_CONVERSION;
  cerr << "Response: " << (const char*)(W2A(spResponseXMLDOM->xml)) << endl;

  CoUninitialize();
#endif
#else
  (void)lod;
  (void)x0;
  (void)y0;
  (void)x1;
  (void)y1;
  (void)slice0;
  (void)slice1;
  (void)dataset_id;
  (void)server;
  (void)ns;
#endif

  return ptr;
}


// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
