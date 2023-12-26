% allclearL2
% Load data structure
% load L2fmri_READINGw
% age = std([24 24 25 22 20 24 25 24 27 29 25 24 22 23 26 25 34 26 23 25 26 25 26 26 32 32 23 30 24 21 21 30 23 23 19])
ismal = L2_str.ismal; % Is subject Malayalam: 1 (for Malayalam) and 0 (for Telugu)
qm = 1:34; qt = 35:68; % Index of Malayalam and Telugu stimuli
qs = 1:10; qd = 11:34; % Index of single and double letter stimuli
[ids, ROIname] = getvoxind(L2_str); % Extracting voxel IDs of various Region of Interest (ROI)-interesection of functional and anatomical regions

%% Plotting mean activity values for different regions
for roi = 1:5 %
    for sub = 1:numel(ismal)
        if ismal(sub); N = qm; NN = qt; else, N = qt; NN = qm; end % Identifying native (N) and non-native (NN) stimuli index
        betas = L2_str.mergedevtbeta{sub};                      % Extracting Beta (or regression weights) for each subject
        if roi == 4; max_vox = 20; elseif roi == 3, max_vox = 200; else, max_vox = Inf; end     % Restricting VWFA definitions upto top 75 voxels
        nvox = min(numel(ids{sub,roi}),max_vox);                % Total number of voxels considered in the analysis
        
        % Averaging betas across single and two letter stimuli; separately for native and nonnative stimuli
        actn(sub,:)  = nanmean(betas(ids{sub,roi}(1:nvox),N));
        actnn(sub,:) = nanmean(betas(ids{sub,roi}(1:nvox),NN));
    end
    
    % Calculating mean and standard error of mean across subjects. Grouping data according to Readers
    meanact = [nanmean(vec(actn(ismal == 0,:))) nanmean(vec(actnn(ismal == 0,:))) ; nanmean(vec(actn(ismal == 1,:))) nanmean(vec(actnn(ismal == 1,:))); nanmean(vec(actn)) nanmean(vec(actnn))];
    semact  = [nansem(nanmean(actn(ismal == 0,:),2)) nansem(nanmean(actnn(ismal == 0,:),2)); nansem(nanmean(actn(ismal == 1,:),2)) nansem(nanmean(actnn(ismal == 1,:),2)); nansem(nanmean(actn,2)) nansem(nanmean(actnn,2))];
    
    % pair-wise statistical test
    P(roi,:) =  [signrank(nanmean(actn(ismal == 0,:),2), nanmean(actnn(ismal == 0,:),2)); signrank(nanmean(actn(ismal == 1,:),2), nanmean(actnn(ismal == 1,:),2)); signrank(nanmean(actn,2), nanmean(actnn,2))];
    
    % Plotting the data
    subplot(1,5,roi); barweb(meanact, semact); title(ROIname{roi}); legend('Known stimuli','Unknown stimuli','Location','Best'); ylabel('Average Betas');  set(gca,'XTickLabel',{'Telugu\newlineReaders','Malayalam\newlineReaders','All subjects'});
end

%%
distmsr = 'spearman'; % 1- r (correlation coefficient) is used as a dissimilarity measure.
% Visual dissimilarity based on model prediction
dpsyTT = L2_str.search.pdouble.dTT; % Telugu    readers on Telugu    bigrams
dpsyMT = L2_str.search.pdouble.dMT; % Malayalam readers on Telugu    bigrams
dpsyTM = L2_str.search.pdouble.dTM; % Telugu    readers on Malayalam bigrams
dpsyMM = L2_str.search.pdouble.dMM; % Malayalam readers on Malayalam bigrams

% Extracting dissimilarity for each subject and each ROI
for rep = 1:1000
    Subjsample = randi(numel(ismal),[numel(ismal),1])';
    for roi = 1:5
        Mcnt = 0; Tcnt = 0;  Tdisn =[]; Tdisnn = []; Mdisn =[]; Mdisnn = [];
        for sub = Subjsample
            if ismal(sub); N = qm; NN = qt; else, N = qt; NN = qm; end
            betas = L2_str.mergedevtbeta{sub};                      % Extracting Beta (or regression weights) for each subject
            if roi == 4; max_vox = 20; elseif roi == 3, max_vox = 200; else, max_vox = Inf; end     % Restricting VWFA definitions upto top 75 voxels
            nvox = min(numel(ids{sub,roi}),max_vox);                % Total number of voxels considered in the analysis
            
            % Calculating pair-wise dissimilarities for single and double letter stimuli
            if ismal(sub)
                Mcnt = Mcnt + 1;
                xx = betas(ids{sub,roi}(1:nvox),N(qd))';  xx(:,isnan(mean(xx))) = []; Mdisn(:,Mcnt)  = pdist(xx,distmsr);
                xx = betas(ids{sub,roi}(1:nvox),NN(qd))'; xx(:,isnan(mean(xx))) = []; Mdisnn(:,Mcnt) = pdist(xx,distmsr);
            else
                Tcnt = Tcnt + 1;
                xx = betas(ids{sub,roi}(1:nvox),N(qd))';  xx(:,isnan(mean(xx))) = []; Tdisn(:,Tcnt)  = pdist(xx,distmsr);
                xx = betas(ids{sub,roi}(1:nvox),NN(qd))'; xx(:,isnan(mean(xx))) = []; Tdisnn(:,Tcnt) = pdist(xx,distmsr);
            end
        end
        
        [R_N(roi,rep), P_N(roi,rep)] = nancorrcoef([zscore(dpsyTT) zscore(dpsyMM)] , [zscore(nanmean(Tdisn,2)) zscore(nanmean(Mdisn,2))]);
        [R_NN(roi,rep), P_NN(roi,rep)] = nancorrcoef([zscore(dpsyMT) zscore(dpsyTM)] , [zscore(nanmean(Mdisnn,2))  zscore(nanmean(Tdisnn,2))]); 

        [Rt(roi,rep), Pt(roi,rep)]   = nancorrcoef(nanmean(Tdisn,2), nanmean(Mdisnn,2));
        [Rm(roi,rep), Pm(roi,rep)] = nancorrcoef(nanmean(Mdisn,2), nanmean(Tdisnn,2));
        % Separately for Telugu and Malayalam
        %              [R_N(roi,rep), P_N(roi,rep)] = nancorrcoef([zscore(dpsyMM)] , [ zscore(nanmean(Mdisn,2))]);
        %              [R_NN(roi,rep), P_NN(roi,rep)] = nancorrcoef([zscore(dpsyMT)] , [zscore(nanmean(Mdisnn,2))]);
        %             [R_N(roi,rep),  P_N(roi,rep)]  = nancorrcoef([zscore(dpsyTT)] , [zscore(nanmean(Tdisn,2))]);
        %             [R_NN(roi,rep), P_NN(roi,rep)] = nancorrcoef([zscore(dpsyTM)] , [zscore(nanmean(Tdisnn,2))]);
        
    end
    disp(rep)
end

p_N = median(P_N,2); p_NN = median(P_NN,2);
figure;  barweb([mean(R_N,2), mean(R_NN,2)]',[std(R_N,[],2), std(R_NN,[],2)]'); colormap(pink); title('Correlation with Behaviour'); legend(ROIname)
set(gca,'Xticklabel',{'Known Stimuli','Unknown Stimuli'}); ylabel('Correlation Coefficient'); 

pt = median(Pt,2); pm = median(Pm,2);
figure; barweb([mean(Rt,2), mean(Rm,2)]',[std(Rt,[],2),std(Rm,[],2)]'); colormap(pink); title('Correlation b/w groups'); legend(ROIname)
set(gca,'Xticklabel',{'Telugu stimuli','Malayalam Stimuli'}); ylabel('Correlation Coefficient'); 

%% ANOVA STATS
for roi = 1 %
    for sub = 1:numel(ismal)
        betas = L2_str.mergedevtbeta{sub};    
        
        % Averaging betas across single and two letter stimuli; separately for native and nonnative stimuli
        actM(sub,1)  = nanmean(nanmean(betas(ids{sub,roi},qm)));
        actT(sub,1) = nanmean(nanmean(betas(ids{sub,roi},qt)));
    end
end
signrank(actT,actM)

% cnt = 1;
% for Sub = 1:35
%     for Script = 1:2
%         script(cnt,1) = Script;
%         sub(cnt,1) = ismal(Sub);
%         if Script == 1
%             act(cnt,1) = actM(Sub); cnt = cnt +1 ;
%         else
%             act(cnt,1) = actT(Sub); cnt = cnt +1 ;
%         end
%     end
% end
% 
% p = anovan(act,{sub, script}');

%% Estimating width of the single letters
load L2_letters.mat
imgid = sort([17 20 25 29 35   18 13 23 33 27 ]);
cnt = 1;
for i = [imgid imgid+36]
    x = mean(L2_str.images{i});
    npix(cnt,1) = numel(find(L2_str.images{i}));
    wid(cnt,1) = numel(find(x));  cnt = cnt + 1;  
end
[mean(wid(1:10)) mean(wid(11:20))]
[mean(npix(1:10)) mean(npix(11:20))]/(300*380)
[std(wid(1:10)) std(wid(11:20))]
[std(npix(1:10)) std(npix(11:20))]/(300*380)


