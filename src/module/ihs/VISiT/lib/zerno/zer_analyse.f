C **************************************************************
c **************************************************************
C ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS ** FENFLOSS **
C **************************************************************
C **************************************************************
      SUBROUTINE ZER_ANALYSE(parti,komp_e,komp_d,lnods,
     *                       zeig,help,
     *                       zer_text,zer_fehler,
     *                       corno_geb,corel_geb,halel_geb,
     *                       recv_geb,send_geb,nach_geb,
     *                       corno_ext,corel_ext,halel_ext,
     *                       recv_ext,send_ext,nach_ext,
     *                       nlapp_ges,ncut_ges)
c
      implicit none

      include 'common.zer'
      include 'mpif.h'
c
      integer parti,komp_e,komp_d,lnods,
     *        zeig,help

      integer corno_geb,corel_geb,halel_geb,
     *        recv_geb,send_geb,nach_geb,
     *        corno_ext,corel_ext,halel_ext,
     *        recv_ext,send_ext,nach_ext,
     *        nlapp_ges,ncut_ges

      integer i,j,k,ipoin,nnn,nhal,ncor,nrecv,
     *        igeb,lu,geb_num,iii,ikn,inach,ifrom

      logical zer_fehler

      character*80 zer_text,comm_text,info_text,zeil_1,zeil_2

      dimension komp_e(nl_kompakt),komp_d(npoin+1),
     *          parti(npoin_max),lnods(nelem_max,nkd),
     *          zeig(npoin_max),help(npoin_max)

      dimension corno_geb(ngebiet),corel_geb(ngebiet),
     *          halel_geb(ngebiet),
     *          recv_geb(ngebiet),send_geb(ngebiet),
     *          nach_geb(ngebiet)

      dimension corno_ext(ndrei),corel_ext(ndrei),
     *          halel_ext(ndrei),
     *          recv_ext(ndrei),send_ext(ndrei),
     *          nach_ext(ndrei)
c     ****************************************************************


c     ****************************************************************
c     INITIALISIERUNGEN:

      zer_fehler=.false.

      do 55 i=1,ngebiet
         corno_geb(i)=0
         corel_geb(i)=0
         halel_geb(i)=0
         recv_geb(i)=0
         send_geb(i)=0
         nach_geb(i)=0
 55   continue

      info_text='Information zur '
      comm_text='Fehler in '

      do 120 i=2,80 
         zeil_1(i-1:i)='*'
         zeil_2(i-1:i)='='
 120  continue
 777  format(1x,A60)
c     ****************************************************************


c     *****************************************************************
c     BESTIMMUNG DER ANZAHL KERN-KNOTEN PRO GEBIET:

      do 101 i=1,ngebiet 
          corno_geb(i)=0
 101  continue

      do 100 i=1,npoin

         corno_geb(parti(i))=corno_geb(parti(i))+1

         if (parti(i).eq.0.or.parti(i).gt.ngebiet) then
            zer_fehler=.true.
         endif

         IF (zer_fehler) THEN

            do 66 iii=1,2

               if (iii.eq.1) then
                  lu=lupro
               else
                  lu=6      
               endif
               write(lu,*)'                   '
               write(lu,777) zeil_2
               write(lu,*)'MELDUNG VON ROUTINE ZER_ANALYSE'
               call char_druck(info_text,zer_text,lu)
               write(lu,*)'Das Partitionierungsfeld besitzt einen '
               write(lu,*)'unzulaessigen Eintrag.                 '
               write(lu,*)'i        =',i                        
               write(lu,*)'parti(i) =',parti(i)               
               write(lu,*)'Die Zerlegung wird fuer ungueltig erklaert.'
               write(lu,777) zeil_2
 66         continue

            goto 999
         ENDIF

 100  continue 
c     *****************************************************************


c     *****************************************************************
c     BESTIMMUNG DER SEND-UND RECEIVE-ANZAHLEN PRO GEBIET:

      do 201 i=1,npoin
         zeig(i)=0
         help(i)=0
 201  continue

      do 203 igeb=1,ngebiet
         send_geb(igeb)=0
 203  continue
      
      do 200 igeb=1,ngebiet 

         nrecv=0
         do 210 ipoin=1,npoin
            if (parti(ipoin).eq.igeb) then

               do 220 j=komp_d(ipoin),komp_d(ipoin+1)-1
                 if (parti(komp_e(j)).ne.igeb) then
                     if (zeig(komp_e(j)).eq.0) then
                        ifrom=parti(komp_e(j))
                        zeig(komp_e(j))=ifrom
                        nrecv=nrecv+1
                        help(nrecv)=komp_e(j)
                     endif
                 endif
 220           continue

            endif
 210     continue

c        Bestimmung der Send-Anzahlen:
         do 260 i=1,nrecv
            ikn=help(i)
            ifrom=zeig(ikn)
            send_geb(ifrom)=send_geb(ifrom)+1
 260     continue

c        Initialisierung:
         do 202 i=1,nrecv
            zeig(help(i))=0
 202     continue

         recv_geb(igeb)=nrecv

 200  continue
c     **************************************************************


c     **************************************************************
c     BESTIMMUNG DER ANZAHL NACHBARBEGIETE:


      do 301 i=1,ngebiet
         zeig(i)=0
         help(i)=0
 301  continue

      do 300 igeb=1,ngebiet 

         nnn=0
         do 310 ipoin=1,npoin
            if (parti(ipoin).eq.igeb) then

               do 320 j=komp_d(ipoin),komp_d(ipoin+1)-1
                  if (parti(komp_e(j)).ne.igeb) then

                      inach=parti(komp_e(j))

                      if (zeig(inach).eq.0) then
c                        Gebiet inach wurde noch nicht gezaehlt:
                         zeig(inach)=1
                         nnn=nnn+1
                         help(nnn)=inach     
                      endif
                  endif
 320           continue

            endif
 310     continue

         do 302 i=1,nnn
            zeig(help(i))=0
 302     continue

         nach_geb(igeb)=nnn  

 300  continue
c     **************************************************************


c     **************************************************************
c     BESTIMMUNG DER ANZAHL KERN- UND HALO-ELEMENTE:

      do 401 i=1,nelem
         zeig(i)=0
         help(i)=0
 401  continue

      do 400 igeb=1,ngebiet
         nhal=0
         ncor=0
         do 410 i=1,nelem
            nnn=0
            do 420 k=1,nkd
               ikn=lnods(i,k)
               if (parti(ikn).eq.igeb) then
                  nnn=nnn+1
               endif
 420        continue 

            if (nnn.ne.0) then
               if (nnn.eq.nkd) then
                  ncor=ncor+1
               else 
                  nhal=nhal+1
               endif
            endif

 410     continue

         corel_geb(igeb)=ncor
         halel_geb(igeb)=nhal
 400  continue
c     **************************************************************

c     **************************************************************
c     BESTIMMUNG DER GESAMTANZAHL AN UEBERLAPPELEMENTEN:

      nlapp_ges=0
      do 450 i=1,nelem
         igeb=parti(lnods(i,1)) 

c        Kontrolle ob alle Knoten zu Gebiet igeb gehoeren:
         nnn=0
         do 460 k=1,nkd
            if (parti(lnods(i,k)).eq.igeb) then
               nnn=nnn+1
            endif
 460     continue 

         if (nnn.ne.nkd) then
            nlapp_ges=nlapp_ges+1
         endif
         
 450  continue
c     **************************************************************


c     **************************************************************
c     BESTIMMUNG DER GESAMTANZAHL AN SCHNITTKANTEN:

      ncut_ges=0
      do 600 ipoin=1,npoin
         geb_num=parti(ipoin)
         do 610 k=komp_d(ipoin),komp_d(ipoin+1)-1
            if (parti(komp_e(k)).ne.geb_num) then
               ncut_ges=ncut_ges+1
            endif 
 610     continue
 600  continue

      ncut_ges=INT(ncut_ges/2)
c     **************************************************************


c     **************************************************************
c     BESTIMMUNG DER EXTREMWERTE:


      corno_ext(1)=+10000000
      corel_ext(1)=+10000000
      halel_ext(1)=+10000000
       recv_ext(1)=+10000000
       send_ext(1)=+10000000
       nach_ext(1)=+10000000

      corno_ext(2)=0.0
      corel_ext(2)=0.0
      halel_ext(2)=0.0
       recv_ext(2)=0.0
       send_ext(2)=0.0
       nach_ext(2)=0.0

      corno_ext(3)=-10000000
      corel_ext(3)=-10000000
      halel_ext(3)=-10000000
       recv_ext(3)=-10000000
       send_ext(3)=-10000000
       nach_ext(3)=-10000000

      do 650 igeb=1,ngebiet
         corno_ext(1)=MIN(corno_ext(1),corno_geb(igeb))
         corel_ext(1)=MIN(corel_ext(1),corel_geb(igeb))
         halel_ext(1)=MIN(halel_ext(1),halel_geb(igeb))

         corno_ext(2)=corno_ext(2)+corno_geb(igeb)
         corel_ext(2)=corel_ext(2)+corel_geb(igeb)
         halel_ext(2)=halel_ext(2)+halel_geb(igeb)

         corno_ext(3)=MAX(corno_ext(3),corno_geb(igeb))
         corel_ext(3)=MAX(corel_ext(3),corel_geb(igeb))
         halel_ext(3)=MAX(halel_ext(3),halel_geb(igeb))



         recv_ext(1)=MIN(recv_ext(1),recv_geb(igeb))
         send_ext(1)=MIN(send_ext(1),send_geb(igeb))
         nach_ext(1)=MIN(nach_ext(1),nach_geb(igeb))

         recv_ext(2)=recv_ext(2)+recv_geb(igeb)
         send_ext(2)=send_ext(2)+send_geb(igeb)
         nach_ext(2)=nach_ext(2)+nach_geb(igeb)

         recv_ext(3)=MAX(recv_ext(3),recv_geb(igeb))
         send_ext(3)=MAX(send_ext(3),send_geb(igeb))
         nach_ext(3)=MAX(nach_ext(3),nach_geb(igeb))
 650  continue

      corno_ext(2)=corno_ext(2)/REAL(ngebiet)
      corel_ext(2)=corel_ext(2)/REAL(ngebiet)
      halel_ext(2)=halel_ext(2)/REAL(ngebiet)

      send_ext(2)=send_ext(2)/REAL(ngebiet)
      recv_ext(2)=recv_ext(2)/REAL(ngebiet)
      nach_ext(2)=nach_ext(2)/REAL(ngebiet)
c     **************************************************************


 999  continue

      return
      end

