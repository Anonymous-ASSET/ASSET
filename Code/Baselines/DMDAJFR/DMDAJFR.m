function [ PD,PF,Precision,F1,AUC,MCC] = DMDAJFR(source, target, parameter)

%%% parameter.lambda;
 %parameter.beta;
 %parameter.noises;
 %parameter.layers;
 
 warning('off');
 X_src = source(:,1:end-1);
 Y_src = source(:,end);
 X_tar = target(:,1:end-1);
 Y_tar = target(:,end);
 parameter.beta = 1;
 
 [Y_tar_pseudo]= Pseudolable(X_src, X_tar, Y_src);
 X_src_new = [];
 X_tar_new = [];

 try
     for i=1:parameter.layers
         try
             [X_src_globalx,X_tar_globalx,W] = GMDA(X_src,Y_src,X_tar,Y_tar_pseudo,parameter);
         catch
             [X_src_globalx,X_tar_globalx,W] = GMDA(single(X_src),single(Y_src),single(X_tar),single(Y_tar_pseudo),parameter);
         end
         try
             [X_src_localx,X_tar_localx,Wc] = LMDA(X_src,Y_src,X_tar,Y_tar_pseudo,parameter);
         catch
             [X_src_localx,X_tar_localx,Wc] = LMDA(single(X_src),single(Y_src),single(X_tar),single(Y_tar_pseudo),parameter);
         end
         
         X_src_new=[X_src_globalx,X_src_localx];
         X_tar_new=[X_tar_globalx,X_tar_localx];
         [Y_tar_pseudo]= Pseudolable(X_src_new,  X_tar_new, Y_src);
     end
     
     model = glmfit(X_src_new, Y_src, 'binomial', 'link', 'logit'); %
     probPos = glmval(model, X_tar_new, 'logit');
     [PD,PF,Precision,F1,AUC,MCC] = Performance(Y_tar(:,end), probPos); % Call Performance()
 catch
    PD=nan;PF=nan;Precision=nan;F1=nan;AUC=nan;MCC=nan;
 end
end

