
// MFCApplication2.h : main header file for the MFCApplication2 application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CMFCApplication2App:
// See MFCApplication2.cpp for the implementation of this class
//

class CMFCApplication2App : public CWinAppEx
{
public:
	CMFCApplication2App();


// Overrides
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// Implementation
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CMFCApplication2App theApp;
