C **************************************************************
C **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE GEO_DIM(geo_name,geo_pfad,nkern_max,
     *                   nlapp_el,nlapp_kn)

      implicit none

      include 'common.zer'

      integer  nkern_max,nlapp_el,nlapp_kn 

      integer i,ipfad,lentb,lu,ip1,ip2,ip3,ip4,iread,
     *        igeb,nkn_ges,nel_ges,kn_max,el_max,nel,nkn,
     *        nel_kern,nel_lapp,luerr,idumm

      logical geo_new_format,geo_old_format,geo_cut_format,
     *        para_format

      character*80 file_name,geo_name,geo_pfad,comment,reihe
      character*100  zeile

      character*4  otto       

      logical fehler,format_read

      parameter(lu=80)
c     **********************************************************


c     **********************************************************
c     INITIALISIERUNGEN:

      nlapp_el=0
      nlapp_kn=0
      nkern_max=0

c     nlapp_el.....Gesamtanzahl Halo-Elemente 
c     nlapp_kn.....Gesamtanzahl Halo-Knoten        
c     nkern_max....Maximum von npoin_ges und nelem_ges
c     **********************************************************



c     **********************************************************
c     DIMENSIONEN DER GEOMETRIE:

      file_name=geo_name
      open(lu,file=file_name,status='old',err=777)
      format_read=.true.
      CALL HEAD_READ(lu,file_name,format_read,reihe)

      npoin=iread(reihe)
      nelem=iread(reihe)
      idumm=iread(reihe)
      idumm=iread(reihe)
      npoin_ges=iread(reihe)
      nelem_ges=iread(reihe)
      knmax_num=iread(reihe)
      elmax_num=iread(reihe)

      nelem_ges=nelem
      npoin_ges=npoin


c     Identifikation des Geometriefiles:
      read(lu,'(A)') zeile
      para_format=.false.
      CALL GEO_IDENT(zeile,file_name,geo_new_format,
     *                  geo_old_format,geo_cut_format,
     *                  para_format)

      if (.not.geo_new_format) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine GEO_DIM'
         write(luerr,*)'Der Geometrie-File besitzt entweder      '
         write(luerr,*)'das alte Flow-Format oder ein            '
         write(luerr,*)'Schnittgitter-Format.                    '
         write(luerr,*)'Diese Zerno-Version funktioniert nur  '
         write(luerr,*)'mit dem neuen Flow-Format.               '
         call erro_ende(myid,parallel,luerr)
      endif

      close(lu)

      if (knmax_num.lt.npoin_ges) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine GEO_DIM'
         write(luerr,*)'Die maximale Knotennummer ist kleiner als die'
         write(luerr,*)'Gesamtanzahl an Knoten.   '
         write(luerr,*)'Maximale Knotennummer:',knmax_num
         write(luerr,*)'Gesamtanzahl Knoten  :',npoin_ges 
         call erro_ende(myid,parallel,luerr)
      endif

      if (elmax_num.ne.nelem_ges) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine GEO_DIM'
         write(luerr,*)'Die maximale Elementnummer muss gleich  der'
         write(luerr,*)'Gesamtanzahl an Elementen sein.'
         write(luerr,*)'Maximale Elementnummer:',elmax_num
         write(luerr,*)'Gesamtanzahl Elemente :',nelem_ges 
         call erro_ende(myid,parallel,luerr)
      endif
c     **********************************************************


c     **********************************************************
c     EINLESEN DER DIMENSIONEN DER PARTITION:

c     Wenn eine Partition eingelesen wird, werden alle Daten auf
c     die Felder lapp_kn_* und lapp_el_* eingelesen. Dies ist
c     wichtig, damit die Knoten- bzw. Element-Reihenfolge  im
c     Ergebnis und Geometrie-File gleich ist. 


      IF (parti_les) THEN

c        Beschriften der Filenamen:
         file_name=geo_pfad
         ipfad=lentb(file_name)
         ip1=ipfad+1
         ip2=ipfad+4
         ip3=ip2+1
         ip4=ip3+3
         file_name(ip1:ip2)='GEO_'

c        Gebietsweises Einlesen der Geometrie-Files:
         do 100 igeb=1,ngebiet

            write(otto,'(i4.4)') igeb
            file_name(ip3:ip4)=otto(1:4)
            open(lu,file=file_name,status='old',err=777)
            format_read=.true.
            CALL HEAD_READ(lu,file_name,format_read,reihe)

            nkn=iread(reihe)
            nel=iread(reihe)
            nel_kern=iread(reihe)
            nel_lapp=iread(reihe)
            nkn_ges=iread(reihe)
            nel_ges=iread(reihe)
            kn_max=iread(reihe)
            el_max=iread(reihe)

            nlapp_el=nlapp_el+nel
            nlapp_kn=nlapp_kn+nkn

c           Identifikation des Geometriefiles:
            read(lu,'(A)') zeile
            para_format=.true. 
            CALL GEO_IDENT(zeile,file_name,geo_new_format,
     *                        geo_old_format,geo_cut_format,
     *                        para_format)

            if (.not.geo_new_format) then
               call erro_init(myid,parallel,luerr)
               write(luerr,*)'Fehler in Routine GEO_DIM'
               write(luerr,*)'Der Geometrie-File von Gebiet ',igeb 
               write(luerr,*)'besitzt entweder das alte Flow-Format'
               write(luerr,*)'oder ein Schnittgitter-Format.       '
               write(luerr,*)'Diese Zerno-Version funktioniert nur  '
               write(luerr,*)'mit dem neuen Flow-Format.               '
               call erro_ende(myid,parallel,luerr)
            endif

            close(lu)

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
              write(luerr,*)'Fehler in Routine GEO_DIM'
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
              write(luerr,*)'igeb            =',igeb        
              call erro_ende(myid,parallel,luerr)
            endif

 100     continue


c        Dimensionen der der Halo- und Kern-Daten:
         nlapp_el=nlapp_el
         nlapp_kn=nlapp_kn
         nkern_max=0                           

c        nlapp_el=nlapp_el-nelem_ges
c        nlapp_kn=nlapp_kn-npoin_ges
c        nkern_max=MAX(npoin_ges,nelem_ges)


      ENDIF
c     **********************************************************


c     **********************************************************
c     FEHLERMELDUNG FUER FALSCH GEOEFFNETE FILES:

      goto 888
 777  continue      
      comment='Fehler beim Oeffnen von File (geo_dim):'
      call erro_init(myid,parallel,luerr)
      call char_druck(comment,file_name,luerr)
      call erro_ende(myid,parallel,luerr)
 888  continue      
c     **********************************************************


      return
      end

