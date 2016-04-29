C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE GRAPH_ELE(lnods,elmat,elmat_adr,elmat_stu,nl_elmat,
     *                     kelem,kelem_adr,nl_kelem,
     *                     help,zeig,mark,graph_all,graph_fla,
     *                     schreiben)
c
      implicit none

      include 'common.zer'

      integer   lnods,elmat,elmat_adr,elmat_stu,kelem,kelem_adr,
     *          nl_kelem,nl_elmat,help,zeig,mark

      integer   i,k,j,ielem,ipoin,adr,el_num,luerr,
     *          nnn,kflag,idummy,nall,nstu,nich,ludru,
     *          nlang,error_number,nnn_adr,nnn_mat

      logical   graph_all,graph_fla,schreiben

      parameter (error_number=-9999999)

      dimension lnods(nelem_max,nkd),kelem(nl_kelem),
     *          elmat_adr(nelem+1),elmat_stu(nelem),elmat(nl_elmat),
     *          kelem_adr(npoin+1),help(npoin),zeig(npoin),mark(npoin)
c     ****************************************************************

c     if (myid.eq.0)write(6,*)'                     '
c     if (myid.eq.0)write(6,*)'Aufruf von GRAPH_ELE '

c     ****************************************************************
c     INITIALISIERUNGEN:

      IF (schreiben) THEN
         do 402 i=1,nelem
            elmat_adr(i+1)=error_number
 402     continue

         do 403 j=1,nl_elmat
            elmat(j)=error_number
 403     continue

         elmat_adr(1)=1
      ENDIF

      nlang=0
c     ****************************************************************


c     ****************************************************************
c     BESTIMMUNG DES ELEMENT-GRAPHEN:

c     Initialisierung:
      do 201 i=1,npoin
        help(i)=0
        zeig(i)=0
        mark(i)=0
 201  continue

      nlang=0
      do 200 ielem=1,nelem
c 
c        Markieren der Knoten von ielem auf Feld mark:
         do 202 k=1,nkd
            mark(lnods(ielem,k))=1
 202     continue

        nall=0

c       Bestimmung der an Element ielem stumpf angrenzenden Elemente:
        nstu=0
        do 210 k=1,nkd
           ipoin=lnods(ielem,k)
           do 220 i=kelem_adr(ipoin),kelem_adr(ipoin+1)-1
              el_num=kelem(i)

              IF (zeig(el_num).eq.0.and.el_num.ne.ielem) THEN
c                Feststellen ob das Element el_num stumpf angrenzt:
                 nnn=0
                 do 230 j=1,nkd
                   if (mark(lnods(el_num,j)).eq.1) then
                      nnn=nnn+1
                   endif
 230             continue

                 if (nnn.eq.INT(nkd/2)) then
c                   Element el_num grenzt stumpf an Element ielem an
                    nstu=nstu+1   
                    nall=nall+1
                    help(nall)=el_num
                    zeig(el_num)=1
                    goto 219
                 endif
              ENDIF

 219       continue
 220       continue
 210    continue

c       Bestimmung der restlichen an Element ielem angrenzenden Elemente
        nich=0
        do 240 k=1,nkd
           ipoin=lnods(ielem,k)
           do 250 i=kelem_adr(ipoin),kelem_adr(ipoin+1)-1
              el_num=kelem(i)

              if (zeig(el_num).eq.0.and.el_num.ne.ielem) then
c                Element el_num grenzt an ielem an und wurde 
c                noch nicht gezaehlt.
                 nich=nich+1   
                 nall=nall+1
                 help(nall)=el_num
                 zeig(el_num)=2
              endif
 250       continue
 240    continue

        if (nall.ne.nstu+nich) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Fehler in Routine GRAPH_ELE'
            write(luerr,*)'Die Summe aus stumpf und nicht stumpf      '
            write(luerr,*)'angrenzender Elemente sit nicht gleich     '
            write(luerr,*)'der Gesamtanzahl angrenzender Elemente.    '
            write(luerr,*)'Element-Nummer:',ielem
            write(luerr,*)'Gesamtanzahl angrenzender Elemente:',nall
            write(luerr,*)'Stumpf       angrenzende  Elemente:',nstu   
            write(luerr,*)'Nicht stumpf angrenzende  Elemente:',nich   
            call erro_ende(myid,parallel,luerr)
        endif                          


        kflag=1
c       Sortieren der stumpf angrenzenden Elemente auf Feld help:
        call isort(help,idummy,nstu,kflag)

c       Sortieren der restilichen Elemente auf Feld help:
        call isort(help(nstu+1),idummy,nich,kflag)
       
 
        IF (schreiben) THEN

           if (graph_all) then
              elmat_adr(ielem+1)=elmat_adr(ielem)+nall
              elmat_stu(ielem)=elmat_adr(ielem)+nstu-1
              do 260 i=1,nall
                 nlang=nlang+1
                 elmat(nlang)=help(i)
                 zeig(help(i))=0
 260          continue
           else if (graph_fla) then
              elmat_adr(ielem+1)=elmat_adr(ielem)+nstu
              do 261 i=1,nstu
                 nlang=nlang+1
                 elmat(nlang)=help(i)
                 zeig(help(i))=0
 261          continue
           endif

        ELSE

           if (graph_all) then
              nlang=nlang+nall
           else if (graph_fla) then
              nlang=nlang+nstu
           endif

        ENDIF

c       Initialisieren der Hilfsfelder:
        do 270 i=1,nall
           zeig(help(i))=0
 270    continue
        do 272 k=1,nkd
            mark(lnods(ielem,k))=0
 272    continue


 200  continue
c     ****************************************************************


c     ****************************************************************
c     DIMENSIONSKONTROLLE:

      IF (schreiben) THEN

         if (nlang.ne.nl_elmat) THEN
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Fehler in Routine GRAPH_ELE'
            write(luerr,*)'Die tatsaechlich geschriebene Anzahl an  '
            write(luerr,*)'an Matrixeintraegen stimmt mit der zuvor '
            write(luerr,*)'bestimmten Anzahl nicht ueberein.        '
            write(luerr,*)'Geschriebene Anzahl:',nlang           
            write(luerr,*)'Bestimmte    Anzahl:',nl_elmat
            call erro_ende(myid,parallel,luerr)
         endif
      ELSE 

        nl_elmat=nlang

      ENDIF
c     ****************************************************************


c     ****************************************************************
c     KONTROLLE OB ALLE FELDER BESCHRIEBEN WURDEN: 

      IF (schreiben) THEN
         nnn_adr=0
         nnn_mat=0
         do 701 i=1,nelem
            if (elmat_adr(i).eq.error_number) then
               nnn_adr=nnn_adr+1
            endif
 701     continue
         if (elmat_adr(nelem+1).eq.error_number) then
            nnn_adr=nnn_adr+1
         endif

         do 702 i=1,nl_elmat
            if (elmat(i).eq.error_number) then
               nnn_mat=nnn_mat+1
            endif
 702     continue

         if (nnn_adr.ne.0.or.nnn_mat.ne.0) then
            call erro_init(myid,parallel,luerr)     
            write(luerr,*)'Fehler in Routine GRAPH_ELE'
            write(luerr,*)'Die Felder zur Beschreibung des   '
            write(luerr,*)'Element-Graphen wurden nicht vollstaendig'
            write(luerr,*)'beschriftet.                              '
            write(luerr,*)'                                          '
            write(luerr,*)'Anzahl fehlender Eintraege in elmat_adr:',
     *                        nnn_adr
            write(luerr,*)'Anzahl fehlender Eintraege in elmat    :',
     *                        nnn_mat
            call erro_ende(myid,parallel,luerr)     
         endif
      ENDIF
c     ****************************************************************

c     ****************************************************************
c     AUSDRUCK:
c
c     IF (schreiben) THEN
c        ludru=lupar
c        write(ludru,*)'                             '
c        write(ludru,*)'AUS GRAPH_ELE                '
c        write(ludru,*)'nl_elmat=',nl_elmat  
c        write(ludru,*)'                             '
c        write(ludru,*)' Nr         Graph:'
c        do 501 i=1,nelem
c          write(ludru,555) i,(elmat_adr(i+1)-elmat_adr(i)),
c    *                        (elmat_stu(i)+1-elmat_adr(i)),
c    *                     (elmat(k),k=elmat_adr(i),elmat_adr(i+1)-1)
c501     continue
c555     format(3(i5,1x),2x,30(i5,1x))
c        write(ludru,*)'                             '
c     ENDIF
c     ****************************************************************

      return
      end

