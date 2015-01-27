C                                                                      |

      subroutine diflin (CCNODE, CCMAX, LKORN, FMESH, NSTEPS, TIME1,
     1                  TIME2, TEMP1, TDIFKO, CCDIFK, D0, TAU, TVERL,
     2                  NT, MVN, NCC, ERRFLG)
C
C  CCNODE = durchschnittliche C-Konzentration des Gefueges
C  CCMAX  = maximale C-Konzentration
C  LKORN  = Korngroesse
C  FMESH  = Anzahl der Unterteilungen in x-Richtung
C  NSTEPS = Anzahl der Zeitschritte
C  TIME1  = Diffusionsstartzeit
C  TIME2  = Diffusionsstopzeit
C  TEMP1  = Temperatur beim Diffusionsstartzeitpunkt
C  TDIFKO = Temperaturwerte der Diffkoeff. (array (100))
C  CCDIFK = C-Konz.-werte der Diffkoeff. (array (100))
C  D0     = Diff.-koef.-werte (array (100,100))
C  TAU    = Taktzeiten aus T_VERLAUF (array)
C  TVERL  = Temperaturverlauf aus T_VERLAUF (array)
C  NT     = Anzahl der Temp.-Werte in Diff.-Koef.-Datei
C  NCC    = Anzahl der C-Konz.-Werte in Diff.-Koef.-Datei
C  ERRFLG = Fehler-Code, = 0 wenn erfolgreich
C           Darf nur vom Hauptprogramm zurÅckgesetzt weren !!!
C
C ======================================================================
      double precision CCNODE, CCMAX, LKORN, TIME1, TIME2, TEMP1,
     1           TDIFKO(100), CCDIFK(100), D0(100,100), TAU(500),
     2           TVERL(5500), cct(1:81,0:100), UD(81),HD(81), OD(81),
     3           RS(81), U(81), D1(80), alpha, CCM, TIMERP, TEMPRP, dt,
     4           dx, gkorn, lx, ly, x, y, fehler, summe
      integer FMESH, NSTEPS, NT, MVN, NCC, ERRFLG, N, i, j
C
C     cct = C-Konz.-Werte ueber der Zeit (array(x,t))
C     TIMERP = Zeit am Rechenpunkt
C     TEMPRP = Temperatur am Rechenpunkt
C     UD     = untere Diagonale (array (max 81))
C     HD     = Hauptiagonale (array (max 81))
C     OD     = obere Diagonale (array (max 81))
C     RS     = Rechte Seite (array (max 81))
C     U      = Loesungsvektor (array (max 81))
C     D1     = Diffusionskoeffizienten (array (max 80))
C     alpha  = Hilsfvariable
C     CCM    = mittlere C-Konz. zur Interpol. von D1
C     gkorn  = Korngrenze
C
       x=CCNODE
      y=CCMAX-x
      if (x.ge.y) then
         lx=LKORN*x/(2*y)
         ly=LKORN/2
      else
         lx=LKORN/2
         ly=LKORN*y/(2*x)
      end if
c      gkorn = LKORN*CCNODE/CCMAX
      gkorn = lx
      print *,'gkorn = ',gkorn
      x=lx+ly
      print *,'lx+ly = ',x
      pause
      dt = (TIME2-TIME1)/NSTEPS
      dx = (lx+ly)/FMESH
      alpha = dt/(dx*dx)
      N = FMESH+1
C
C     Anfangs-Kohlenstoffkonzentration
C     ------------------------------------------------------------------
       do 10,i = 1,N
         if ((dx*i).lt.gkorn) then
            cct(i,0)=ccmax
         end if
         if ((dx*i).ge.gkorn) then
            cct(i,0)=0
         end if
10    continue
C
C     Diffusionsrechnung
      open (10,file='CCOUT')
C
      do 40,j = 0,NSTEPS
         TIMERP = TIME1+j*dt
C
C       Temp.-Bestimmung zum Zeitpunkt dt*j mittels  Linearinterpolation
C        ---------------------------------------------------------------
         call ipol1p (TAU, TVERL, TIMERP, TEMPRP, MVN)
C
         do 20,i = 1,FMESH
C
C       Bestimmung der mittleren C-Konz. zwischen 2 aufeinanderfolgenden
C           Rechenpunkten zur Bestimmung von D1[i*1/2]
C           ------------------------------------------------------------
c            print *,i,j,cct(i,j)
            CCM = (cct(i,j)+cct(i+1,j))/2
            call ipol2p (TDIFKO,CCDIFK, D0, TEMPRP, CCM, D1(i), NT, NCC)
c            print *, TEMPRP,CCM,D1(i)
c            print *,CCM,D1(i)
20       continue
         do 30,i=2,N-1
            UD(i) = -alpha*D1(i-1)
            HD(i) = 2+alpha*(D1(i-1)+D1(i))
            OD(i) = -alpha*D1(i)
      RS(i)= alpha*D1(i-1)*cct(i-1,j)+(2-alpha*(D1(i-1)+D1(i)))*cct(i,j)
     1                 +alpha*D1(i)*cct(i+1,j)
30       continue
         HD(1) = 2+2*alpha*D1(1)
         OD(1) = -2*alpha*D1(1)
         RS(1) = (2-2*alpha*D1(1))*cct(1,j)+2*alpha*D1(1)*cct(2,j)
         UD(N) = -2*alpha*D1(N-1)
         HD(N) = 2+2*alpha*D1(N-1)
         RS(N) = 2*alpha*D1(N-1)*cct(N-1,j)+(2-2*alpha*D1(N-1))*cct(N,j)

C        C-Konz.-Bestimmung fuer naechsten Zeitschritt
C        ---------------------------------------------------------------
         call tridag (UD,HD,OD,RS,U,N)
         fehler=1
32       summe=0
         do 35,i=1,N
            U(i)=U(i)/fehler
c            U(i)=U(i)/((fehler-1)*2+1)
            cct(i,j+1) = U(i)
            if (cct(i,j+1).gt.CCMAX) cct(i,j+1)=CCMAX
c            print *,cct(i,j+1)
            summe=summe+cct(i,j+1)
35       continue
         summe=summe/N
         fehler=summe/CCNODE
c         print *,summe,fehler
c         pause
c         if ((fehler.lt.0.999).or.(fehler.gt.1.001)) goto 32
40    continue
C     ==================================================================
      write (10,*) 'C-Konzentration nach Diff.-Rechnung'
      do 50,i=1,N
          write (10,*) cct(i,NSTEPS)
50    continue
      close (10)
C
      RETURN
      END
C
