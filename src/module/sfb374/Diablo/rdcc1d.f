c   liest aus 'CCOUT' C-Konzentrationen nach der Diff.-Rechnung
c   ablauf vergleiche rdcout.f
c
      subroutine rdc1dd (CC, CCMIN, CCMAX, FMESH, ERRFLG)
c
      double precision CC(100),CCMIN, CCMAX
      integer ERRFLG, i, FMESH
      character strg1*40
c
      open (10,file='CCOUT',ERR=60)
20    read (10,'(a36)',ERR=60,END=70) strg1
      if (strg1.ne.' C-Konzentration nach Diff.-Rechnung') goto 20
      do 50,i=1,FMESH+1
	 read (10,*,ERR=60,END=70) cc(i)
50    continue
      goto 70
60    PRINT*
      PRINT*,'Fehler beim Einlesen der CC-Werte nach Diff.-Rechnung'
      close (10)
      return
70    close (10)
      return
      end
