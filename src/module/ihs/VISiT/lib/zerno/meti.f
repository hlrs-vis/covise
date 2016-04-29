C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE METI(komp_e,komp_d,lnods,coord_num,knpar,
     *                parti,nparti,zer_zeig,ndat_max,
     *                zeig,help,folg,
     *                corno_geb,corel_geb,recv_geb,send_geb,
     *                halel_geb,nach_geb,
     *                corno_ext,corel_ext,recv_ext,send_ext,
     *                halel_ext,nach_ext)
  
      implicit none     

      include 'common.zer'

      integer  komp_e,komp_d,lnods,coord_num,knpar,
     *         parti,nparti,zer_zeig,ndat_max,zer_memory,
     *         zeig,help,folg

      integer corno_geb,corel_geb,recv_geb,send_geb,halel_geb,
     *        nach_geb

      integer corno_ext,corel_ext,recv_ext,send_ext,halel_ext,
     *        nach_ext

      integer i,k,weiflag,forflag,optio(5),
     *        ncut_kway,ncut_recu,
     *        edge_wei,node_wei

      integer parti_error,nlappel_ges,ncuting_ges,ncut_ges,nlapp_ges,
     *        help_dat,help_num,help_int,iwahl,nwahl

      integer lu,iii,luerr,nhelp_max,ipar,igeb,icom,lentb,
     *        nkriterium

      logical zer_fehler

      character*80 zer_text,zer_name,zeil_text(10),
     *             zeil_1,zeil_2,zeil_3,zeil_4,wahl_name

      parameter (nhelp_max=10)

      dimension lnods(nelem_max,nkd),parti(npoin_max,nparti),
     *          coord_num(npoin_max),knpar(npoin_max),
     *          komp_d(npoin+1),komp_e(nl_kompakt),
     *          zeig(npoin_max),help(npoin_max),folg(npoin_max)

      dimension zer_zeig(ndat_max),zer_memory(nhelp_max)

      dimension corno_geb(ngebiet,nparti),corel_geb(ngebiet,nparti),
     *          halel_geb(ngebiet,nparti),
     *          recv_geb(ngebiet,nparti),send_geb(ngebiet,nparti),
     *          nach_geb(ngebiet,nparti)
     
      dimension corno_ext(ndrei,nparti),corel_ext(ndrei,nparti),
     *          halel_ext(ndrei,nparti),
     *          recv_ext(ndrei,nparti),send_ext(ndrei,nparti),
     *          nach_ext(ndrei,nparti)

      dimension zer_name(nhelp_max),parti_error(nhelp_max),
     *          nlappel_ges(nhelp_max),ncuting_ges(nhelp_max),
     *          help_dat(nhelp_max),help_num(nhelp_max),
     *          help_int(nhelp_max)
c     *****************************************************************


c     ****************************************************************
c     DIMENSIONSKONTROLLEN:

      if (nparti.gt.nhelp_max) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine METI'
         write(luerr,*)'Die Dimension der Hilfsfelder ist zu klein.'
         write(luerr,*)'Benoetigt    :',nparti
         write(luerr,*)'Dimensioniert:',nhelp_max                   
         write(luerr,*)'Parameter nhelp_max in Routine METI erhoehen.' 
         call erro_ende(myid,parallel,luerr)
      endif
c     ****************************************************************


c     ****************************************************************
c     INITIALISIERUNGEN:

      do 120 i=2,80 
         zeil_1(i-1:i)='*'
         zeil_2(i-1:i)='z'
         zeil_3(i-1:i)='-'
         zeil_4(i-1:i)='!'
 120  continue
 777  format(1x,A70)
 666  format(1x,A)


      do 101 i=1,npoin
        do 102 k=1,nparti
           parti(i,k)=0
 102    continue
 101  continue

      weiflag=0
      forflag=1
      optio(1)=1
      optio(2)=3
      optio(3)=1
      optio(4)=1
      optio(5)=0
      edge_wei=0
      node_wei=0

c     Beschriftung der Zerlegungsnamen:
      zer_name(1)='Zerlegung mit METIS_PartGraphKway'
      zer_name(2)='Zerlegung mit METIS_PartGraphVKway'
      zer_name(3)='Zerlegung mit METIS_PartGraphRecursive'

      zer_memory(1)=0
      zer_memory(2)=0
      zer_memory(3)=0
c     ****************************************************************

c        write(lupar,*)'AUSDRUCK AUS METI  nl_kompakt=',nl_kompakt
c        do 733 i=1,npoin
c          write(lupar,744) coord_num(i),komp_d(i+1)-komp_d(i),
c    *                 (coord_num(komp_e(k)),k=komp_d(i),komp_d(i+1)-1)
c733     continue
c744     format(i4,1x,i3,1x,50(i3,1x))

c     ****************************************************************
c     ZERLEGUNGEN UND ANALYSE:

      do 100 ipar=1,nparti

         if (myid.eq.0.and.zer_zeig(ipar).eq.1) then
            zer_text=zer_name(ipar)
            icom=lentb(zer_text)
            write(6,*)'                                    '
            write(6,666) zer_text(1:icom)
         endif

         IF (ipar.eq.1.and.zer_zeig(ipar).eq.1) THEN
            CALL METIS_EstimateMemory(npoin,komp_d,komp_e,1,2,
     *                                zer_memory(ipar))
            CALL METIS_PartGraphKway(npoin,komp_d,komp_e,
     *                               edge_wei,node_wei,
     *                               weiflag,forflag,ngebiet,
     *                               optio,
     *                               ncut_kway,parti(1,ipar))

         ELSE IF (ipar.eq.2.and.zer_zeig(ipar).eq.1) THEN
            zer_memory(ipar)=-999999
            CALL METIS_PartGraphVKway(npoin,komp_d,komp_e,
     *                               edge_wei,node_wei,
     *                               weiflag,forflag,ngebiet,
     *                               optio,
     *                               ncut_kway,parti(1,ipar))

         ELSE IF (ipar.eq.3.and.zer_zeig(ipar).eq.1) THEN
            CALL METIS_EstimateMemory(npoin,komp_d,komp_e,1,1,
     *                                zer_memory(ipar))
            CALL METIS_PartGraphRecursive(npoin,komp_d,komp_e,   
     *                                    edge_wei,node_wei,      
     *                                    weiflag,forflag,ngebiet,
     *                                    optio,
     *                                    ncut_recu,parti(1,ipar))
         ENDIF

         zer_text=zer_name(ipar)
         if (zer_zeig(ipar).eq.1) then
            CALL ZER_ANALYSE(parti(1,ipar),komp_e,komp_d,lnods,
     *                       zeig,help,
     *                       zer_text,zer_fehler,
     *                       corno_geb(1,ipar),corel_geb(1,ipar),
     *                       halel_geb(1,ipar),
     *                       recv_geb(1,ipar),send_geb(1,ipar),
     *                       nach_geb(1,ipar),
     *                       corno_ext(1,ipar),corel_ext(1,ipar),
     *                       halel_ext(1,ipar),
     *                       recv_ext(1,ipar),send_ext(1,ipar),
     *                       nach_ext(1,ipar),
     *                       nlapp_ges,ncut_ges)
         else
           zer_fehler=.true.
           nlapp_ges=10000000
           ncut_ges =10000000
         endif

         nlappel_ges(ipar)=nlapp_ges
         ncuting_ges(ipar)=ncut_ges 

         if (zer_fehler) then
            parti_error(ipar)=1
         else
            parti_error(ipar)=0
         endif

         if (myid.eq.0.and.zer_zeig(ipar).eq.1) then
            write(6,*)'Zerlegung beendet '
         endif
 100  continue
c     ****************************************************************



c     ****************************************************************
c     AUSWAHL EINER ZERLEGUNG:

c     Feststellen ob  mindestens eine gueltige Zerlegung existiert:
      zer_fehler=.true.  
      do 401 ipar=1,nparti
         if (parti_error(ipar).eq.0) then
            zer_fehler=.false.
         endif 
 401  continue
      if (zer_fehler) then
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine METI  '
         write(luerr,*)'Alle Zerlegungen sind ungueltig.'
         call erro_ende(myid,parallel,luerr)
      endif


c     Die Auswahl erfolgt ueber die mittleren Anzahlen der Zerlegungen:
c     Auswahlreihenfolge:
c             - kleinste mittlere Receive-Anzahl
c             - kleinste mittlere Send-Anzahl
c             - kleinste mittlere Nachbar-Anzahl 
c             - kleinste Gesamtanzahl an Ueberlappelementen


c     INITIALISIERUNGEN FUER DIE AUSWAHL:
      iwahl=0
      nwahl=nparti
      do 402 i=1,nwahl
         help_num(i)=i
 402  continue


      nkriterium=4
      do 400 iii=1,nkriterium

          do 410 i=1,nwahl
             ipar=help_num(i)
             if (iii.eq.1) then
                help_dat(i)=recv_ext(2,ipar)
             else if (iii.eq.2) then
                help_dat(i)=send_ext(2,ipar)
             else if (iii.eq.3) then
                help_dat(i)=nach_ext(2,ipar)
             else if (iii.eq.4) then
                help_dat(i)=nlappel_ges(ipar)
             endif
 410      continue

          CALL CHOICE(help_dat,help_num,help_int,nwahl,
     *                parti_error,nparti)

          if (nwahl.eq.1) then
c            Es gibt genau einen minimalen Eintrag auf help_dat 
c            -> Beste Zerlegung gefunden

             iwahl=help_num(1)
             goto 444
          endif 

          if (iii.eq.nkriterium) then
c            Es gibt Zerlegungen die auch beim letzten Kriterium 
c            gleich sind. -> Es wird einfach eine dieser 
c            Zerlegungen genommen

             iwahl=help_num(1)
          endif

 400  continue
 444  continue

      IF (iwahl.eq.0.or.iwahl.gt.nparti) THEN
         call erro_init(myid,parallel,luerr)
         write(luerr,*)'Fehler in Routine METI'
         write(luerr,*)'Es wurde keine Zerlegung ausgwaehlt obwohl'
         write(luerr,*)'es zulaessige Zerlegungen gibt.           '
         call erro_ende(myid,parallel,luerr)
      ELSE
         wahl_name=zer_name(iwahl)
         icom=lentb(wahl_name)
         wahl_name(icom+1:icom+15)=' wird verwendet'   

      ENDIF
c     ****************************************************************


c     ****************************************************************
c     KOPIEREN DER AUSGEWAEHLTEN ZERLEGUNG: 

      do 200 i=1,npoin
         knpar(i)=parti(i,iwahl)
 200  continue

c     write(lupar,*)'Partitionen'
c     do 201 i=1,npoin
c        write(lupar,222) coord_num(i),(parti(i,k),k=1,nparti)
c        write(lupar,222) coord_num(i),parti(i,iwahl)
c201  continue
c222  format(i7,3x,3(i4,1x))
c     ****************************************************************


c     ****************************************************************
c     AUSGABE DER GEBIETS-DATEN:

      zeil_text(1)='Gebiet    '
      zeil_text(2)='npoin_kern'
      zeil_text(3)='nelem_kern'
      zeil_text(4)='nelem_lapp'
      zeil_text(5)='nrecv     '
      zeil_text(6)='nsend     '
      zeil_text(7)='nach      '

      do 500 ipar=1,nparti

         zer_text=zer_name(ipar)
         icom=lentb(zer_text)

         if (myid.eq.0) then

            do 501 iii=1,1
               if (iii.eq.1) then
                  lu=lupro
               else
                  lu=6
               endif
               write(lu,*)'                                   ' 
               write(lu,777) zeil_1
               write(lu,666) zer_text(1:icom)          
               write(lu,666) zeil_3(1:icom)
               write(lu,*)'                                       '

               if (parti_error(ipar).eq.0) then

                  write(lu,554) (zeil_text(k),k=1,7)
                  do 550 igeb=1,ngebiet 
                     write(lu,556) igeb,corno_geb(igeb,ipar),
     *                                  corel_geb(igeb,ipar),
     *                                  halel_geb(igeb,ipar),
     *                                   recv_geb(igeb,ipar),
     *                                   send_geb(igeb,ipar),
     *                                   nach_geb(igeb,ipar)
 550              continue
                  write(lu,*)'                                   ' 
                  write(lu,*)'Gesamtanzahl Ueberlappelemente     :',
     *                        nlappel_ges(ipar)
                  write(lu,*)'Gesamtanzahl Schnittkanten im Graph:',
     *                        ncuting_ges(ipar)

               else                          
                  if (zer_zeig(ipar).eq.1) then
                     write(lu,*)' Die Zerlegung ist ungueltig !!! '
                  endif
               endif

               write(lu,777) zeil_1
 501        continue

         endif

 500  continue

 554  format(1x,A6,3x,3(A10,2x),2x,2(A5,4x),A4)                 
 556  format(2x,i4,6x,2(i7,5x),i7,3x,2(i7,2x),2x,i4)              
c     ****************************************************************


c     ****************************************************************
c     AUSGABE DER EXTREMWERTE DER GEBIETS-DATEN:

      zeil_text(1)='          '
      zeil_text(2)='npoin_kern'
      zeil_text(3)='nelem_kern'
      zeil_text(4)='nelem_lapp'
      zeil_text(5)='nrecv     '
      zeil_text(6)='nsend     '
      zeil_text(7)='nach      '

      if (myid.eq.0) then

         do 301 iii=1,2

         if (iii.eq.1) then
            lu=lupro
         else if (iii.eq.2) then
            lu=6
         endif

         write(lu,*)'                                       '
         write(lu,777) zeil_1

         do 300 ipar=1,nparti

            zer_text=zer_name(ipar)
            icom=lentb(zer_text)

            if (parti_error(ipar).eq.0) then
               write(lu,666) zer_text(1:icom)          
               write(lu,666) zeil_3(1:icom)
               write(lu,334) (zeil_text(k),k=1,7)
               write(lu,337) 'MAXIMAL:',corno_ext(3,ipar),
     *                                  corel_ext(3,ipar),
     *                                  halel_ext(3,ipar),
     *                                   recv_ext(3,ipar),
     *                                   send_ext(3,ipar),
     *                                   nach_ext(3,ipar)
               write(lu,337) 'MITTEL :',corno_ext(2,ipar),
     *                                  corel_ext(2,ipar),
     *                                  halel_ext(2,ipar),
     *                                   recv_ext(2,ipar),
     *                                   send_ext(2,ipar),
     *                                   nach_ext(2,ipar)
               write(lu,337) 'MINIMAL:',corno_ext(1,ipar),
     *                                  corel_ext(1,ipar),
     *                                  halel_ext(1,ipar),
     *                                   recv_ext(1,ipar),
     *                                   send_ext(1,ipar),
     *                                   nach_ext(1,ipar)

               write(lu,*)'Gesamtanzahl Ueberlappelemente     :',
     *                     nlappel_ges(ipar)
               write(lu,*)'Gesamtanzahl Schnittkanten im Graph:',
     *                     ncuting_ges(ipar)
               write(lu,*)'                                       '
            else 
               if (zer_zeig(ipar).eq.1) then
                  write(lu,*)' Die Zerlegung ist ungueltig !!! '
                  write(lu,*)'                                       '
               endif
            endif
            if (ipar.eq.nparti) then
               icom=lentb(wahl_name)
               write(lu,666) wahl_name(1:icom)          
            endif

 300     continue               

         write(lu,777) zeil_1
 301     continue

      endif
 334  format(1x,A6,3x,3(A10,2x),2x,2(A5,4x),A4)                 
 337  format(1x,A8,3x,2(i7,5x),i7,3x,2(i7,2x),2x,i4)              
 338  format(1x,A8,27x,i7)
c     ****************************************************************

c     ****************************************************************
c     SPEICHER DER METIS-ROUTNEN:

      do 302 iii=1,2

         if (iii.eq.1) then
            lu=lupro
         else if (iii.eq.2) then
            lu=6
         endif

         write(lu,*)'                                       '
         write(lu,777) zeil_1
         write(lu,*)'SPEICHER DER METIS-ROUTINEN:           '
         write(lu,*)'---------------------------            '
         write(lu,*)'Memory fuer METIS_PartGraphKway      [ kBytes ]:',
     *               zer_memory(1)
c        write(lu,*)'Memory fuer METIS_PartGraphVKway     [ kBytes ]:',
c    *               zer_memory(2)
         write(lu,*)'Memory fuer METIS_PartGraphRecursive [ kBytes ]:',
     *               zer_memory(3)
         write(lu,777) zeil_1
         write(lu,*)'                                       '

 302  continue
c     ****************************************************************

      return
      end

