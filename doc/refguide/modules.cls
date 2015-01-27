%%
%% This is file `modules.cls',
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
%\ProvidesClass{book}
\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagetrue
\newif\if@openright
\newif\if@mainmatter \@mainmattertrue



%%%===============================================	
%%% Papiergroessen	
%%%===============================================	
\if@compatibility\else
\DeclareOption{covisepaper}
   {\setlength\paperheight {230mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi

%%%===============================================	
%%% Fontgroessen	
%%%===============================================	
\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}


%%%===============================================	
%%% andere globale Parameter der Klasse	
%%%===============================================	
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility
\@openrighttrue
\else
\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}


%%%===============================================	
%%% Defaultgroessen	
%%%===============================================	
\ExecuteOptions{covisepaper,10pt,twoside,onecolumn,final,openright}



\ProcessOptions
\input{bk1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\setlength\parindent{0em}
\setlength\parskip{1ex plus.2ex minus.2ex}
%\setlength\parskip{0\p@ \@plus \p@}
\renewcommand\baselinestretch{}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
%\setcounter{topnumber}{5}
%\renewcommand\topfraction{.9}
%\setcounter{bottomnumber}{5}
%\renewcommand\bottomfraction{.0}
%\setcounter{totalnumber}{5}
%\renewcommand\textfloatsep{0.}
%\renewcommand\textfraction{0.}
%\renewcommand\floatpagefraction{0.}
%\setcounter{dbltopnumber}{2}
%\renewcommand\dbltopfraction{.7}
%\renewcommand\dblfloatpagefraction{.5}


%%%===============================================	
%%% heading and footing fuer Seiten	
%%%===============================================	
\def\ps@headings{%
	 \let\@evenfoot\@empty%
    \let\@oddfoot\@empty%    
	 \def\@oddhead {\vbox{\hfill\rightmark\hspace{3ex}\thepage \vskip 1ex \hrule}}%				 				 
    \def\@evenhead{\vbox{\thepage\hspace{3ex}\leftmark\hfil   \vskip 1ex \hrule}}%
    \let\@mkboth\markboth
	 
  \def\chaptermark##1{%
                 %%\markboth {\bf{%
    \markboth {\bf{%
      \ifnum \c@secnumdepth >\m@ne
        \if@mainmatter
          %\@chapapp\ \thechapter. \ %
        \fi
      \fi
      ##1}}{}}%
  \def\sectionmark##1{%
                %\markright {\bf{%
    \markright {\bf{%
      \ifnum \c@secnumdepth >\z@
        \thesection. \ %
      \fi
      ##1}}}}

\def\ps@myheadings{%
	 \let\@evenfoot\@empty%
    \let\@oddfoot\@empty%    
	 \def\@oddhead {\vbox{\hfill\rightmark\hspace{3ex}\thepage \vskip 1ex \hrule}}%				 				 
    \def\@evenhead{\vbox{\thepage\hspace{3ex}\leftmark\hfil   \vskip 1ex \hrule}}%
    \let\@mkboth\markboth
    \let\@mkboth\@gobbletwo
    \let\chaptermark\@gobble
    \let\sectionmark\@gobble
    }
	 
	 
%%%==================================================	
%%% neue Kommandos fuer Titelseite und nächste Seite
%%%==================================================	

\newcommand*{\@extratitle}{}
\newcommand{\extratitle}[1]{\gdef\@extratitle{#1}}
\newcommand*{\@titlehead}{}
\newcommand{\titlehead}[1]{\gdef\@titlehead{#1}}
\newcommand*{\@subject}{}
\newcommand*{\subject}[1]{\gdef\@subject{#1}}
\newcommand*{\@publishers}{}
\newcommand{\publishers}[1]{\gdef\@publishers{#1}}
\newcommand*{\@uppertitleback}{}
\newcommand{\uppertitleback}[1]{\gdef\@uppertitleback{#1}}
\newcommand*{\@lowertitleback}{}
\newcommand{\lowertitleback}[1]{\gdef\@lowertitleback{#1}}
\newcommand*{\@dedication}{}
\newcommand{\dedication}[1]{\gdef\@dedication{#1}}
\newcommand*{\next@tpage}{\clearpage\thispagestyle{empty}}

\if@titlepage
    \newcommand*\maketitle[1][1]{\begin{titlepage}%
    \setcounter{page}{#1}
    %\let\footnotesize\small
    %\let\footnoterule\relax
    %\let\footnote\thanks
    %\renewcommand*\thefootnote{\@fnsymbol\c@footnote}%
    %\let\@oldmakefnmark\@makefnmark
    %\def\@makefnmark{\rlap\@oldmakefnmark}
	 \thispagestyle{empty}
    \unitlength1cm
    \begin{picture}(15,17)
    %\put(-4,-1){\includegraphics[scale=0.85]{title.eps}}
    \put(0,10){\makebox(15,2)[t]{\Huge{COVISE Module Reference Guide}}}
    \put(0,6){\makebox(15,2)[t]{\large{(September 2003)}}}
    \put(0,-3){\line(1,0){15}}
    \end{picture}
	 
    \if@twoside\next@tpage
        \noindent\begin{minipage}[t]{\textwidth}
        \@uppertitleback
        \end{minipage}\par
        \vfill
		  
        \noindent\begin{minipage}[b]{\textwidth}
        \@lowertitleback \par
    	  %{\@date \par}
        \end{minipage}
    \fi
    \end{titlepage}
	 
	 
    \setcounter{footnote}{0}%
    \global\let\thanks\relax
    \global\let\maketitle\relax
    \global\let\@thanks\@empty
    \global\let\@author\@empty
    \global\let\@date\@empty
    \global\let\@title\@empty
    \global\let\@extratitle\@empty
    \global\let\@titlehead\@empty
    \global\let\@subject\@empty
    \global\let\@publishers\@empty
    \global\let\@uppertitleback\@empty
    \global\let\@lowertitleback\@empty
    \global\let\@dedication\@empty
    \global\let\author\relax
    \global\let\title\relax
    \global\let\extratitle\relax
    \global\let\titlehead\relax
    \global\let\subject\relax
    \global\let\publishers\relax
    \global\let\uppertitleback\relax
    \global\let\lowertitleback\relax
    \global\let\dedication\relax
    \global\let\date\relax
    \global\let\and\relax}
\else
     \newcommand*\maketitle[1][1]{\par
       \@tempcnta=#1\relax\ifnum\@tempcnta=1\else
         \ClassWarning{scrbook}
                      {Optional argument of \string\maketitle\space ignored
                       at\MessageBreak
                       notitlepage-mode}\fi
       \begingroup
         \renewcommand*\thefootnote{\@fnsymbol\c@footnote}%
         \let\@oldmakefnmark\@makefnmark
         \def\@makefnmark{\rlap\@oldmakefnmark}
         \if@twocolumn
           \ifnum \col@number=\@ne
             \@maketitle
           \else
             \twocolumn[\@maketitle]%
           \fi
         \else
           \newpage
           \global\@topnum\z@   % Prevents figures from going at top of page.
           \@maketitle
         \fi
         \thispagestyle{plain}\@thanks
       \endgroup
       \setcounter{footnote}{0}%
       \let\thanks\relax
       \let\maketitle\relax
       \let\@maketitle\relax
       \global\let\@thanks\@empty
       \global\let\@author\@empty
       \global\let\@date\@empty
       \global\let\@title\@empty
       \global\let\@extratitle\@empty
       \global\let\@titlehead\@empty
       \global\let\@subject\@empty
       \global\let\@publishers\@empty
       \global\let\@uppertitleback\@empty
       \global\let\@lowertitleback\@empty
       \global\let\@dedication\@empty
       \global\let\author\relax
       \global\let\title\relax
       \global\let\extratitle\relax
       \global\let\titlehead\relax
       \global\let\subject\relax
       \global\let\publishers\relax
       \global\let\uppertitleback\relax
       \global\let\lowertitleback\relax
       \global\let\dedication\relax
       \global\let\date\relax
       \global\let\and\relax}
\fi
\def\@maketitle{%
    \clearpage
    \let\footnote\thanks
    \ifx\@extratitle\@empty \else
        \noindent\@extratitle \next@tpage \if@twoside \null\next@tpage \fi
    \fi
    \ifx\@titlehead\@empty \else
        \noindent\begin{minipage}[t]{\textwidth}
        \@titlehead
        \end{minipage}\par
    \fi
    \null
    \vskip 2em%
    \begin{center}%
    \ifx\@subject\@empty \else
        {\Large \@subject \par}
        \vskip 1.5em
    \fi
    {\huge \@title \par}%
    \vskip 1.5em%
    {\Large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\Large \@date \par}%
    \vskip \z@ \@plus 1em
    {\Large \@publishers \par}
    \ifx\@dedication\@empty \else
        \vskip 2em
        {\Large \@dedication \par}
    \fi
  \end{center}%
  \par
  \vskip 2em}
  
  
  
\newcommand*\chaptermark[1]{}
\setcounter{secnumdepth}{2}
\newcounter {part}
\newcounter {chapter}
\newcounter {section}%[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thechapter {\@arabic\c@chapter}
%\renewcommand \thesection {\thechapter.\@arabic\c@section}
\renewcommand \thesection {\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection .\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\newcommand\@chapapp{\chaptername}
\newcommand\frontmatter{%
    \cleardoublepage
  \@mainmatterfalse
  \pagenumbering{roman}}
\newcommand\mainmatter{%
    \cleardoublepage
  \@mainmattertrue
  \pagenumbering{arabic}}
\newcommand\backmatter{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \@mainmatterfalse}
\newcommand\part{%
  \if@openright
    \cleardoublepage
  \else
    \clearpage
  \fi
  \thispagestyle{plain}%
  \if@twocolumn
    \onecolumn
    \@tempswatrue
  \else
    \@tempswafalse
  \fi
  \null\vfil
  \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >-2\relax
       \huge\bfseries \partname~\thepart
       \par
       \vskip 20\p@
     \fi
     \Huge \bfseries #2\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \normalfont
     \Huge \bfseries #1\par}%
    \@endpart}
\def\@endpart{\vfil\newpage
              \if@twoside
                \null
                \thispagestyle{empty}%
                \newpage
              \fi
              \if@tempswa
                \twocolumn
              \fi}
\newcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{myheadings}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                       \if@mainmatter
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                       \else
                         \addcontentsline{toc}{chapter}{#1}%
                       \fi
                    \else
                      \addcontentsline{toc}{chapter}{#1}%
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
						  
%% Randversuche
\setlength\topmargin {-1.5cm}
\setlength\oddsidemargin {0.5cm}
\setlength\evensidemargin {0.5cm}
\setlength\textheight {17cm}
\setlength\textwidth {15cm}
\setlength\footskip {0cm}
\setlength\headsep {0.5cm}
\setlength\headheight {0.7cm}
		  
%%%===============================================	
%%% Ueberschriften fuer Chapter Seite	
%%%===============================================	
%\def\@makechapterhead#1{%
  %\vspace*{20\p@}%
  %{\parindent \z@ \raggedright \normalfont
    %\ifnum \c@secnumdepth >\m@ne
    %  \if@mainmatter
    %    \hspace{-1cm}{\huge\bfseries \@chapapp\space \thechapter}
    %    \par\nobreak
    %    \vskip 20\p@
    %  \fi
    %\fi
    %\interlinepenalty\@M
    %\Huge \bfseries \hspace{-1cm}\thechapter{#1}\par\nobreak
    %\vskip 40\p@
  %}}
\def\@makechapterhead#1{%
\vspace*{0\p@}%
{\parindent \z@ \raggedright \normalfont
  \Huge \bfseries \hspace{-1.5cm} \thechapter \space \space {#1} \par\nobreak
  \vskip 30\p@
}}
  
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
					  
					  
%%%===============================================	
%%% Ueberschriften fuer  Content, Figures, Tables	
%%%===============================================			  
%\def\@makeschapterhead#1{%
%  \vspace*{50\p@}%
%  {\parindent \z@ \raggedright
%    \normalfont
%    \interlinepenalty\@M
%    \Huge \bfseries  #1\par\nobreak
%    \vskip 40\p@
 % }}	
  				  
\def\@makeschapterhead#1{%
  \vspace*{0\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \Huge \bfseries  #1\par\nobreak
    \vskip 30\p@
  }}
  
  
  
\newcommand\section{\@startsection {section}{1}{\z@ \hspace{-1.2cm}}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex }%
                                   {2.3ex \@plus .2ex }%
                                   {\normalfont\Large\bfseries}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-2ex\@plus -.2ex \@minus -.2ex}%
                                     {1ex \@plus .2ex}%
                                     {\normalfont\large\bfseries}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-2ex\@plus -.2ex \@minus -.2ex}%
                                     {1ex \@plus .2ex}%
                                     {\normalfont\normalsize\bfseries}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {-2ex \@plus.2ex \@minus.2ex}%
                                    {-1ex}%
                                    {\normalfont\normalsize\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {2ex \@plus.2ex \@minus .2ex}%
                                       {-1ex}%
                                      {\normalfont\normalsize\bfseries}}
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\if@compatibility
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \cleardoublepage
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
\newcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \gdef\@chapapp{\appendixname}%
  \gdef\thechapter{\@Alph\c@chapter}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset {equation}{chapter}
\renewcommand\theequation
  {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@equation}
\newcounter{figure}[chapter]
\renewcommand \thefigure
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@figure}
\def\fps@figure{htbp}
\def\ftype@figure{1}
\def\ext@figure{lof}

%%%%  figure unterschrift fett
\def\fnum@figure{\bf\figurename~\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand \thetable
     {\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}

%%%%%  tabellen unterschrift fett
\def\fnum@table{\bf\tablename~\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
%=======================================
\setcounter{tocdepth}{2}
\newcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{%
      		\bf{\contentsname}}{\bf{\contentsname}}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
    \endgroup
  \fi}
\newcommand*\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\newcommand*\l@section{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand*\l@subsection{\@dottedtocline{2}{3.8em}{3.2em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{10em}{5em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{12em}{6em}}
\newcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\bf\listfigurename}%
              {\bf\listfigurename}}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{%
          \bf\listtablename}{\bf\listtablename}}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\chapter*{\bibname
        \@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\@makeschapterhead{\indexname}]%
                \@mkboth{\MakeUppercase\indexname}%
                        {\MakeUppercase\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\@addtoreset{footnote}{chapter}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{Contents}
\newcommand\listfigurename{Figures}
\newcommand\listtablename{Tables}
\newcommand\bibname{Bibliography}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\chaptername{Chapter}
\newcommand\appendixname{Appendix}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{headings}
\pagenumbering{arabic}
\if@twoside
\else
  %\raggedbottom
  \flushbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi


%Macros for context free grammars
\def\termquote#1{{\tt `#1'}}
\def\trm#1{{\tt `#1'}}
\def\nt#1{{\it #1}}
\newcounter{synt}
%Left part of a rule
\newcommand{\ruleleft}[1]{#1 \> $\rightarrow$ \> }
%Simple right part of a rule
\newcommand{\ruleright} [1]{#1 \> \   \\}

%right part of a rule to be continued in the next line
\newcommand{\rulerighta}[1]{#1 \> $\mid$\ \ \\}
%same nut with comments
\newcommand{\rulerightc}[2]{#1 \> $\mid$\ \  {\small\it#2} \\}
%Simple right part of a rule with comments
\newcommand{\rulerightd}[2]{#1 \> \   \ {\small\it#2} \\}
%indent the text to come in a way that it belongs to the rule
%in the line before
\newcommand{\contrule}{\> \>}

%define the widest rule to be
\newcommand{\rulescheme}[2]{#1 \= $\rightarrow$ \= #2 \= $\mid$ \kill}

\newenvironment{syntax}
{\stepcounter{synt}\begin{tabbing}}{\end{tabbing} }



\endinput
%%
%% End of file `covise.cls'.
