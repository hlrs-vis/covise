C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE PAR_LES(coord,coord_num,coord_zeig,
     *                   lnods,lnods_num,lnods_zeig,
     *                   kern_kn,kern_kn_adr,
     *                   lapp_kn,lapp_kn_adr,lapp_kn_proz,
     *                   kern_el,kern_el_adr,
     *                   lapp_el,lapp_el_adr,lapp_el_proz,
     *                   nkern_max,nlapp_el,nlapp_kn,
     *                   knpar,elpar,
     *                   geo_pfad)
c
      implicit none

      include 'common.zer'

      integer    coord_num,coord_zeig,lnods,lnods_num,lnods_zeig

      integer    kern_kn,kern_kn_adr,
     *           lapp_kn,lapp_kn_adr,lapp_kn_proz,
     *           kern_el,kern_el_adr,
     *           lapp_el,lapp_el_adr,lapp_el_proz,
     *           nkern_max,nlapp_el,nlapp_kn,
     *           knpar,elpar

      integer    nu_geo_max

      integer    ispalt,i,j,k,lu,knoten(8),iproz,iread,mod_num,
     *           lp,kn_num,luerr,el_num,join_num,iadr,icom,
     *           lentb,nspalt_max

      integer    nkn,nkn_kern,nkn_lapp,nkn_ges,
     *           nel,nel_kern,nel_lapp,nel_ges,
     *           nnn,nnn_kern,nnn_lapp,
     *           kn_max,el_max,igeb,
     *           ip1,ip2,ip3,ip4,ipfad

      logical    fehler,format_read

      real       coord,dd_geo_max,we_geo_max,
     *           prozent,ddd,www,tol,dd,werte(20)

      character*80 file_name,geo_pfad,comment,reihe,zeil_1,zeil_2

      character*72 zeil_text(6),spal_text(10)
      character*4  otto                         

      parameter(tol=1.e-06)
      parameter(lu=66,nspalt_max=512)
      
      dimension coord(npoin_max,ncd),coord_num(npoin_max),
     *          coord_zeig(knmax_num),lnods_zeig(elmax_num),
     *          lnods(nelem_max,nkd),lnods_num(nelem_max)

      dimension kern_kn(nkern_max),kern_kn_adr(ngebiet+1),
     *          kern_el(nkern_max),kern_el_adr(ngebiet+1)

      dimension lapp_el(nlapp_el),lapp_el_adr(ngebiet+1),
     *          lapp_kn(nlapp_kn),lapp_kn_adr(ngebiet+1),
     *          lapp_el_proz(nlapp_el),lapp_kn_proz(nlapp_kn) 

      dimension knpar(npoin_max),elpar(nelem_max)

      dimension dd_geo_max(nspalt_max),nu_geo_max(nspalt_max),
     *          we_geo_max(nspalt_max)          
c     ****************************************************************


c     Wenn eine Partition eingelesen wird, werden alle Daten auf
c     die Felder lapp_kn_* und lapp_el_* eingelesen. Dies ist
c     wichtig, damit die Knoten- bzw. Element-Reihenfolge  im
c     Ergebnis und Geometrie-File gleich ist. 


c     ****************************************************************
c     DIMENSIONSKONTROLLE:

      if (ngebiet.gt.nspalt_max) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine PAR_LES'
         write(luerr,*)'Die Dimension nspalt_max ist zu klein.     '
         write(luerr,*)'Benoetigt        ngebiet=',ngebiet    
         write(luerr,*)'Dimensioniert nspalt_max=',nspalt_max 
         call erro_ende(myid,parallel,luerr)
      endif
c     ****************************************************************


c     ****************************************************************
c     Initialisieren der Hilfsfelder:

      do 10 ispalt=1,nspalt_max
         dd_geo_max(ispalt)=-1.e+10
         we_geo_max(ispalt)=0.0        
         nu_geo_max(ispalt)=0        
  10  continue

      do 11 i=1,npoin_max
         coord_num(i)=0
         do 12 k=1,ncd
            coord(i,k)=0.0
 12      continue
 11   continue

      do 13 i=1,nelem_max
         lnods_num(i)=0
         do 14 k=1,nkd
            lnods(i,k)=0.0
 14      continue
 13   continue

      do 15 i=1,knmax_num
         coord_zeig(i)=0    
 15   continue
      do 16 i=1,elmax_num
         lnods_zeig(i)=0    
 16   continue
c     ****************************************************************


c     **********************************************************
c     EINLESEN DER PARTITIONIERTEN GEOMETRIE:

c     Beschriften der Filenamen:
      file_name=geo_pfad
      ipfad=lentb(file_name)
      ip1=ipfad+1
      ip2=ipfad+4
      ip3=ip2+1
      ip4=ip3+3
      file_name(ip1:ip2)='GEO_'


c     GEBIETSWEISES EINLESEN DER GEOMETRIE-FILES:
      nkn_ges=0
      nel_ges=0
      kn_max=0
      el_max=0

      nkn_kern=0
      nkn_lapp=0
      nel_kern=0
      nel_lapp=0
      kern_kn_adr(1)=1
      lapp_kn_adr(1)=1
      kern_el_adr(1)=1
      lapp_el_adr(1)=1

      do 100 igeb=1,ngebiet

         write(otto,'(i4.4)') igeb
         file_name(ip3:ip4)=otto(1:4)
         open(lu,file=file_name,status='unknown',err=777)
         format_read=.true.
         CALL HEAD_READ(lu,file_name,format_read,reihe)

         nkn=iread(reihe)
         nel=iread(reihe)

         write(6,*)'Einlesen der Daten von Gebiet ',igeb

c        EINLESEN DER KNOTENDATEN: 
         nnn_kern=0
         nnn_lapp=0
         do 120 i=1,nkn

            read(lu,*) kn_num,(werte(k),k=1,3),mod_num,iproz,join_num
            iadr=abs(join_num)

            if (iproz.eq.igeb) then
               nnn_kern=nnn_kern+1
               nkn_kern=nkn_kern+1
            endif
            nnn_lapp=nnn_lapp+1
            nkn_lapp=nkn_lapp+1
            lapp_kn(nkn_lapp)=iadr
            lapp_kn_proz(nkn_lapp)=iproz

            if (coord_num(iadr).ne.0.and.coord_num(iadr).ne.kn_num) then
               call erro_init(myid,parallel,luerr)
               write(luerr,*)'Fehler in Routine PAR_LES'
               write(luerr,*)'In Gebiet ',igeb,' gibt es Knoten'
               write(luerr,*)'mit derselben Zusammenbaunummer  '
               write(luerr,*)'aber unterschiedlichen Knotennummern.'
               write(luerr,*)'Zusammenbaunummer:',iadr              
               write(luerr,*)'1. Kontennummer  :',coord_num(iadr)   
               write(luerr,*)'2. Kontennummer  :',kn_num            
               call erro_ende(myid,parallel,luerr)
            endif

            IF (coord_num(iadr).eq.0) THEN

c              Knoten wurde noch nicht eingelesen -> Schreiben
               coord_num(iadr)=kn_num
               coord_zeig(kn_num)=iadr
               do 140 k=1,ncd
                  coord(iadr,k)=werte(k)
 140           continue
               nkn_ges=MAX(nkn_ges,iadr)
               kn_max=MAX(kn_max,kn_num)

            ELSE IF (coord_num(iadr).ne.0) THEN

c               Bestimmung der Abweichung zu geschriebenem Knoten:
                do 130 k=1,ncd 
   	           dd=abs(coord(iadr,k)-werte(k))
                   if (dd.gt.dd_geo_max(k)) then
                      dd_geo_max(k)=dd
                      nu_geo_max(k)=kn_num
                      we_geo_max(k)=werte(k)
                   endif
 130            continue

            ENDIF
 120     continue

         kern_kn_adr(igeb+1)=kern_kn_adr(igeb)+0
         lapp_kn_adr(igeb+1)=lapp_kn_adr(igeb)+nnn_lapp



c        EINLESEN DER ELEMENTDATEN:
         nnn_kern=0
         nnn_lapp=0
         do 200 i=1,nel
            read(lu,*) el_num,(knoten(k),k=1,nkd),mod_num,
     *                                     iproz,join_num
         
            iadr=abs(join_num)

            if (iproz.eq.igeb) then
               nnn_kern=nnn_kern+1
               nel_kern=nel_kern+1
            endif
            nnn_lapp=nnn_lapp+1
            nel_lapp=nel_lapp+1
            lapp_el(nel_lapp)=iadr
            lapp_el_proz(nel_lapp)=iproz

            if (lnods_num(iadr).ne.0.and.lnods_num(iadr).ne.el_num) then
               call erro_init(myid,parallel,luerr)
               write(luerr,*)'Fehler in Routine PAR_LES'
               write(luerr,*)'In Gebiet ',igeb,' gibt es Elemente'
               write(luerr,*)'mit derselben Zuasammenbaunummer  '
               write(luerr,*)'aber unterschiedlichen Elementnummern.'
               write(luerr,*)'Zusammenbaunummer:',iadr              
               write(luerr,*)'1. Elementnummer  :',lnods_num(iadr)   
               write(luerr,*)'2. Elementnummer  :',el_num            
               call erro_ende(myid,parallel,luerr)
            endif

c           VERGLEICH MIT BEREITS GESCHRIEBENEN DATEN:
            if (lnods_num(iadr).ne.0) then
                do 220 k=1,nkd 
                   if (lnods(iadr,k).ne.coord_zeig(knoten(k))) then
                      call erro_init(myid,parallel,luerr)
                      write(luerr,*)'In Gebiet ',igeb,' gibt es '
                      write(luerr,*)'Elemente mit derselben  '
                      write(luerr,*)'Zusammenbaunummer aber '
                      write(luerr,*)'unterschiedlichen Knotennummern.'
                      write(luerr,*)'Elementnummer    :',el_num   
                      write(luerr,*)'Zusammenbaunummer:',iadr
                      write(luerr,*)'1. Knotennummern:'
                      write(luerr,222)(coord_num(lnods(iadr,j)),j=1,nkd)
                      write(luerr,*)'2. Knotennummern:'
                      write(luerr,222)(knoten(j),j=1,nkd)
                      call erro_ende(myid,parallel,luerr)
                   endif
 220            continue
            endif

c           KONTROLLE DER KNOTEN-NUMMERN:
            do 231 k=1,nkd       
               kn_num=knoten(k)
               if (kn_num.gt.knmax_num.or.coord_zeig(kn_num).le.0) THEN
                  call erro_init(myid,parallel,luerr)
                  write(luerr,*)'FEHLER IN ELEMENT-KNOTEN-LISTE    '
                  write(luerr,*)'Die angegebene Knotennummer ', kn_num
                  write(luerr,*)'existiert nicht im Geometrie-File bzw.'
                  write(luerr,*)'besitzt keine Koordinaten.       '
                  call erro_ende(myid,parallel,luerr)
               endif
               knoten(k)=coord_zeig(kn_num)
 231        continue


c           SCHRIEBEN DER ELEMENT-DATEN:
            do 230 k=1,nkd
               lnods(iadr,k)=knoten(k)
 230        continue
            lnods_num(iadr)=el_num
            lnods_zeig(el_num)=iadr
            nel_ges=MAX(nel_ges,iadr)
            el_max=MAX(el_max,el_num)

 200     continue

         kern_el_adr(igeb+1)=kern_el_adr(igeb)+0
         lapp_el_adr(igeb+1)=lapp_el_adr(igeb)+nnn_lapp

         close(lu)
 100  continue

 222  format(8(i7,1x))
c     **********************************************************


c     **********************************************************
c     DIMENSIONS-KONTROLLE:

      do 150 i=1,npoin
         if (coord_num(i).eq.0) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Fehler in Routine PAR_LES'
            write(luerr,*)'Es wurden nicht alle Knoten eingelesen'
            write(luerr,*)'i             =',i                          
            write(luerr,*)'coord_num(i)  =',coord_num(i)               
            write(luerr,*)'npoin=        =',npoin
            write(luerr,*)'knmax_num     =',knmax_num                  
            call erro_ende(myid,parallel,luerr)
         endif
 150  continue

      do 160 i=1,nelem
         if (lnods_num(i).eq.0) then
            call erro_init(myid,parallel,luerr)
            write(luerr,*)'Fehler in Routine PAR_LES'
            write(luerr,*)'Es wurden nicht alle Elemente eingelesen'
            write(luerr,*)'i             =',i                          
            write(luerr,*)'lnods_num(i)  =',lnods_num(i)               
            write(luerr,*)'nnelem        =',nelem
            write(luerr,*)'elmax_num     =',elmax_num                  
            call erro_ende(myid,parallel,luerr)
         endif
 160  continue

      nnn=lapp_kn_adr(ngebiet+1)-lapp_kn_adr(1)
      if (nnn.ne.nlapp_kn) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine PAR_LES'
         write(luerr,*)'Die eingelesene Anzahl an Knoten      '
         write(luerr,*)'stimmt mit der in Routine GEO_DIM bestimmten '
         write(luerr,*)'Anzahl nicht ueberein.                       '
         write(luerr,*)'Eingelesene Anzahl:',nnn          
         write(luerr,*)'Bestimmte   Anzahl:',nlapp_kn   
         call erro_ende(myid,parallel,luerr)
      endif

      nnn=lapp_el_adr(ngebiet+1)-lapp_el_adr(1)
      if (nnn.ne.nlapp_el) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine PAR_LES'
         write(luerr,*)'Die eingelesene Anzahl an Elementen   '
         write(luerr,*)'stimmt mit der in Routine GEO_DIM bestimmten '
         write(luerr,*)'Anzahl nicht ueberein.                       '
         write(luerr,*)'Eingelesene Anzahl:',nnn          
         write(luerr,*)'Bestimmte   Anzahl:',nlapp_el   
         call erro_ende(myid,parallel,luerr)
      endif

      if (nkn_kern.ne.npoin_ges) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine PAR_LES'
         write(luerr,*)'Die Summe aller Kern-Knoten muss gleich der '
         write(luerr,*)'Knotenanzahl der Gesamtgeometrie sein.     '
         write(luerr,*)'Knotenanzahl der Gesamtgeometrie:',npoin_ges
         write(luerr,*)'Summe aller Kernknoten          :',nnn
         call erro_ende(myid,parallel,luerr)
      endif

      if (nel_kern.ne.nelem_ges) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine PAR_LES'
         write(luerr,*)'Die Summe aller Kern-Elemente muss gleich der '
         write(luerr,*)'Elementanzahl der Gesamtgeometrie sein.     '
         write(luerr,*)'Elementanzahl der Gesamtgeometrie:',nelem_ges
         write(luerr,*)'Summe aller Kern-Elemente        :',nnn
         call erro_ende(myid,parallel,luerr)
      endif
c     **********************************************************


c     **********************************************************
c     BELEGUNG DER PARTITIONIERUNGSFELDER:

      do 501 i=1,npoin_max
         knpar(i)=0
 501  continue
      do 502 i=1,nelem_max
         elpar(i)=0
 502  continue
   
      do 500 igeb=1,ngebiet 
         do 510 i=lapp_kn_adr(igeb),lapp_kn_adr(igeb+1)-1
            knpar(lapp_kn(i))=igeb
 510     continue

         do 520 i=lapp_el_adr(igeb),lapp_el_adr(igeb+1)-1
            elpar(lapp_el(i))=igeb
 520     continue

 500  continue
c     **********************************************************



c     **********************************************************
c     FEHLER-MELDUNGEN:

      fehler=.false.
      if (nkn_ges.ne.npoin) then
         fehler=.true.
      endif
      if (nel_ges.ne.nelem) then
         fehler=.true.
      endif
      if (kn_max.ne.knmax_num) then
         fehler=.true.
      endif
      if (el_max.ne.elmax_num) then
         fehler=.true.
      endif

      if (fehler) then
        call erro_init(myid,parallel,luerr)
        write(luerr,*)'Fehler in Routine PAR_LES'
        write(luerr,*)'Die Dimensionen der partitionierten '
        write(luerr,*)'Geometrie sind nicht konsistent mit dem'
        write(luerr,*)'eingelesenen Geometrie-File.           '
        write(luerr,*)'Dimensionen der eingelesenen Geometrie:'
        write(luerr,*)'npoin_ges       =',npoin_ges
        write(luerr,*)'nelem_ges       =',nelem_ges
        write(luerr,*)'knmax_num       =',knmax_num
        write(luerr,*)'elmax_num       =',elmax_num
        write(luerr,*)'Dimensionen der partitionierten Geometrie:'
        write(luerr,*)'npoin_ges       =',nkn_ges  
        write(luerr,*)'nelem_ges       =',nel_ges  
        write(luerr,*)'knmax_num       =',kn_max   
        write(luerr,*)'elmax_num       =',el_max   
        call erro_ende(myid,parallel,luerr)
      endif
c     **********************************************************



c     ****************************************************************
c     AUSDRUCK DER ABWEICHUNGEN:        

      do 601 i=2,80  
         zeil_1(i-1:i)='*'
         zeil_2(i-1:i)='-'
 601  continue

      spal_text(1)=' Abweichung'
      spal_text(2)='Absolutwert'
      spal_text(3)='    Prozent'
      spal_text(4)='  KnNr.'

      zeil_text(1)='X'
      zeil_text(2)='Y'
      zeil_text(3)='Z'

      DO 400 i=1,2
            if (i.eq.1) then
              lp=6
            else if (i.eq.2) then
              lp=lupro
            endif

            write(lp,*)                   
            write(lp,55) zeil_1
            comment='Maximale Koordinaten-Abweichungen:'
            icom=lentb(comment)
            write(lp,66) comment(1:icom)
            write(lp,66) zeil_2(1:icom)
            write(lp,77) (spal_text(k),k=1,4)
            do 401 k=1,ncd      

c              Berechnung der prozentualen Abweichung:
               ddd=abs(dd_geo_max(k))
               www=abs(we_geo_max(k))
               if (ddd.lt.tol) then
                  prozent=0.0 
               else if (www.lt.tol.and.ddd.gt.tol) then
                  prozent=-99
               else 
                  prozent=ddd*100.0/www
               endif

    	       write(lp,88) zeil_text(k),dd_geo_max(k),
     *                                    we_geo_max(k),
     *                                    prozent,
     *                                    nu_geo_max(k)
 401        continue
            write(lp,55) zeil_1
            write(lp,*)                   

 400  continue           

 55   format(1x,A70)
 66   format(1x,A)
 77   format(3x,4(5x,A11),3x,A7)
 88   format(1x,A1,2x,3(f15.6,1x),2x,i7,1x,A1)
c     ****************************************************************


c     **********************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:

      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File (par_les):'
      call erro_init(myid,parallel,luerr)
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     **********************************************************


      return
      end

