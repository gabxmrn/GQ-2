function[pnul,pneg,ppos]=computationproportions(pvec,signpvec,nbsimul)
%pvec=vector of pvalues
%signpvec=sign of the coefficients associated with the pvalues (1 if
%positive and -1 if negative)
%nbsimul=number of simulations for the bootstrap

%computation of the null proportion using the boostrap approach of Storey
%(2002)
n=size(pvec,1);
R=[0.50:0.05:0.95]';
nbtest=size(R,1);
pnultot=zeros(nbtest,1);
for i=1:nbtest,
    W=sum(pvec>=R(i,1));
    pnultot(i,1)=(W/n)/(1-R(i,1));
end
minp=min(pnultot);
bootpnultot=zeros(nbtest,nbsimul);
for j=1:nbsimul,
    B=pvec(unidrnd(n,1,n),:);
    for i=1:nbtest,
        W=sum(B>=R(i,1));
        bootpnultot(i,j)=W/((1-R(i,1))*n);
    end
end
difference=bootpnultot-ones(nbtest,nbsimul)*minp;
squared=difference.^2;
mse=(1/nbsimul)*sum(squared')';
lambda=min(mse);
index=find(mse==lambda);
optimalR=R(index(1,1),1);
pnul=pnultot(index(1,1),1);
if pnul<0,
   pnul=0;
elseif pnul>1,
    pnul=1;
end

%computation of the negative and positive proportions
selecn=find(signpvec<0);
pvecneg=pvec(selecn,1);
pnega=sum(pvecneg<optimalR)/n;
pneg=pnega-pnul*optimalR/2;
if pneg<0,
   pneg=0;
end
selecp=find(signpvec>0);
pvecpos=pvec(selecp,1);
pposa=sum(pvecpos<optimalR)/n;
ppos=pposa-pnul*optimalR/2;
if ppos<0,
   ppos=0;
end
