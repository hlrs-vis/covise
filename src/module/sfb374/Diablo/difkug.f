C                                                                      |
      subroutine difkug (CCNODE, CCMAX, LKORN, FMESH, NSTEPS, TIME1,
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
     1         TDIFKO(100), CCDIFK(100), D0(100,100), TAU(500),
     2         TVERL(500), cct(1:81,0:100), UD(81),HD(81), OD(81),
     3         RS(81), U(81), a(80), D1(80), alpha, CCM, TIMERP, TEMPRP,
     4         dt, dr, gkorn, Rges, x, y, summe, fehler
      integer FMESH, NSTEPS, NT, MVN, NCC, ERRFLG, N, i, j
C
C     cct    = C-Konz.-Werte ueber der Zeit (array(x,t))
C     TIMERP = Zeit am Rechenpunkt
C     TEMPRP = Temperatur am Rechenpunkt
C     UD     = untere Diagonale (array (max 81))
C     HD     = Hauptiagonale (array (max 81))
C     OD     = obere Diagonale (array (max 81))
C     RS     = Rechte Seite (array (max 81))
C     U      = Loesungsvektor (array (max 81))
C     D1     = Diffusionskoeffizienten (array (max 80))
C     a      = a=r^2*D1 (array (max 80))
C     aa  = Hilsfvariable
C     CCM    = mittlere C-Konz. zur Interpol. von D1
C     gkorn  = Korngrenze
C
      TEMPRP=0.0
      dt = (TIME2-TIME1)/NSTEPS
      N = FMESH+1
      x=CCNODE
      y=CCMAX-x
      if (x.gt.y) then
        if (CCNODE.le.7*CCMAX/8) then
          Rges=(LKORN/2)/(1-(1-(CCNODE/CCMAX))**(1/3.))
          gkorn=Rges-LKORN/2
        else
          gkorn=LKORN/2
          Rges=(LKORN/2)/((1-(CCNODE/CCMAX))**(1/3.))
        end if
c        print *,gkorn, Rges
c        pause
        dr = Rges/FMESH
C
C       Anfangs-Kohlenstoffkonzentration
C       ----------------------------------------------------------------
        do 10,i = 1,N
          if ((dr*i).lt.gkorn) then
            cct(i,0)=0
c             print *,cct(i,0)
          end if
          if ((dr*i).ge.gkorn) then
            cct(i,0)=CCMAX
c            print *,cct(i,0)
          end if
10      continue
c        pause
C
      else
C
C     grenzfall:lx=ly fuer CCMAX=8*CCNODE
C
        if (CCNODE.le.CCMAX/8) then
          gkorn=LKORN/2
          Rges=(LKORN/2)*((CCMAX/CCNODE)**(1/3.))
        else
          Rges=(LKORN/2)/(1-((CCNODE/CCMAX)**(1/3.)))
          gkorn=Rges-LKORN/2
        end if
        dr = Rges/FMESH
c        print *,gkorn, Rges, dr
c        pause
C
C        Anfangs-Kohlenstoffkonzentration
C        ---------------------------------------------------------------
         do 15,i = 1,N
            if ((dr*i).le.gkorn) then
               cct(i,0)=ccmax
c            print *,cct(i,0)
            end if
            if ((dr*i).gt.gkorn) then
               cct(i,0)=0
c            print *,cct(i,0)
            end if
15       continue
      end if
      if (CCNODE.eq.CCMAX) then
	 print *,'Maximale C-Konzentration vorhanden'
	 print *,'keine Diffrechnung noetig !'
	 return
      end if
      alpha = dt/(dr**4)
C
C     Diffusionsrechnung
C
      open (10,file='CCOUT')
c      open (22,file='probe')
c      write (22,*) FMESH,NSTEPS
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
C           Rechenpunkten zur Bestimmung von D1[i-1/2]
C           ------------------------------------------------------------
c            print *,i,j,cct(i,j)
            CCM = (cct(i,j)+cct(i+1,j))/2
            call ipol2p (TDIFKO,CCDIFK, D0, TEMPRP, CCM, D1(i), NT, NCC)
C
C           ------------------------------------------------------------
C           Bestimmung von a(r) : a(r)=r[i-1/2]^2*D1[i-1/2]
C
            a(i)=(i-0.5)*(i-0.5)*dr*dr*D1(i)
c            print *, TEMPRP,CCM,D1(i)
c            print *,CCM,D1(i)
20       continue
         do 30,i=2,N-1
	    UD(i) = -alpha*a(i-1)
c            write (22,*) 'UD(',i,') = ',UD(i)
            HD(i) = 2*(i-1)*(i-1)+alpha*(a(i-1)+a(i))
c            write (22,*) 'HD(',i,') = ',HD(i)
            OD(i) = -alpha*a(i)
c            write (22,*) 'OD(',i,') = ',OD(i)
            RS(i)= alpha*a(i-1)*cct(i-1,j)
     1             +(2*(i-1)*(i-1)-alpha*(a(i-1)+a(i)))*cct(i,j)
     2             +alpha*a(i)*cct(i+1,j)
c               write (22,*) 'RS(',i,') = ',RS(i)
30       continue
         HD(1) = 2*alpha*a(1)
         OD(1) = -2*alpha*a(1)
         RS(1) = -2*alpha*a(1)*cct(1,j)+2*alpha*a(1)*cct(2,j)
         UD(N) = -2*alpha*a(N-1)
         HD(N) = 2*(N-1)*(N-1)+2*alpha*a(N-1)
         RS(N) = 2*alpha*a(N-1)*cct(N-1,j)
     1          +(2*(N-1)*(N-1)-2*alpha*a(N-1))*cct(N,j)
C
C        C-Konz.-Bestimmung fuer naechsten Zeitschritt
C        ---------------------------------------------------------------
         call tridag (UD,HD,OD,RS,U,N)
c         write (22,*) 'TIMESTEP = ',j
	fehler=1
c        print *,ccnode
32      summe=0
         do 35,i=1,N
            U(i)=U(i)/fehler
            cct(i,j+1) = U(i)
            if (cct(i,j+1).gt.CCMAX) cct(i,j+1)=CCMAX
            summe=summe+cct(i,j+1)*(i**3-(i-1)**3)
35       continue
       summe=summe/(FMESH**3)
       fehler=summe/CCNODE
c       print *,summe,fehler
c       pause
c       if ((fehler.lt.0.999).or.(fehler.gt.1.001)) goto 32
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
