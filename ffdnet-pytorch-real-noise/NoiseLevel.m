% https://github.com/csjunxu/PGPD_Offline_BID/blob/master/NoiseLevel.m

function [nlevel,th,num] = NoiseLevel(img,patchsize,decim,conf,itr)

if( ~exist('itr', 'var') )
    itr = 3;
end

if( ~exist('conf', 'var') )
    conf = 1-1E-6;
end

if( ~exist('decim', 'var') )
    decim = 0;
end

if( ~exist('patchsize', 'var') )
    patchsize = 7;
end


kh = [-1/2,0,1/2];
imgh = imfilter(img,kh,'replicate');
imgh = imgh(:,2:size(imgh,2)-1,:);
imgh = imgh .* imgh;

kv = kh';
imgv = imfilter(img,kv,'replicate');
imgv = imgv(2:size(imgv,1)-1,:,:);
imgv = imgv .* imgv;

Dh = my_convmtx2(kh,patchsize,patchsize);
Dv = my_convmtx2(kv,patchsize,patchsize);
DD = Dh'*Dh+Dv'*Dv;
r = rank(DD);

Dtr = trace(DD);
tau0 = gaminv(conf,double(r)/2, 2.0 * Dtr / double(r));

%{
eg = eig(DD);
tau0 = gaminv(conf,double(r)/2, 2.0 * eg(patchsize*patchsize));
%}

for cha=1:size(img,3)
	X = im2col(img(:,:,cha),[patchsize patchsize]);
	Xh = im2col(imgh(:,:,cha),[patchsize patchsize-2]);
	Xv = im2col(imgv(:,:,cha),[patchsize-2 patchsize]);
    
	Xtr = sum(vertcat(Xh,Xv));

	if( decim > 0 )
	    XtrX = vertcat(Xtr,X);
	    XtrX = sortrows(XtrX')';
	    p = floor(size(XtrX,2)/(decim+1));
	    p = [1:p] * (decim+1);
	    Xtr = XtrX(1,p);
	    X = XtrX(2:size(XtrX,1),p);
	end

	%%%%% noise level estimation %%%%%
    tau = Inf;
    if( size(X,2) < size(X,1) )
        sig2 = 0;
    else    
        cov = X*X'/(size(X,2)-1);
        d = eig(cov);
        sig2 = d(1);
    end
    	    
	for i=2:itr
	%%%%% weak texture selectioin %%%%%
	    tau = sig2 * tau0;
	    p = (Xtr<tau);
	    Xtr = Xtr(:,p);
	    X = X(:,p);
       
	    %%%%% noise level estimation %%%%%
        if( size(X,2) < size(X,1) )
            break;
        end
	    cov = X*X'/(size(X,2)-1);
	    d = eig(cov);
	    sig2 = d(1);	    
	end

	nlevel(cha) = sqrt(sig2);
	th(cha) = tau;
	num(cha) = size(X,2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function T = my_convmtx2(H, m, n)
s = size(H);
T = zeros((m-s(1)+1) * (n-s(2)+1), m*n);

k = 1;
for i=1:(m-s(1)+1)
 for j=1:(n-s(2)+1)
  
  for p=1:s(1)
   T(k,(i-1+p-1)*n+(j-1)+1:(i-1+p-1)*n+(j-1)+1+s(2)-1) = H(p,:);
  end
  
  k = k + 1;
 end
end

