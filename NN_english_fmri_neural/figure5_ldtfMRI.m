if ~exist('L2_str'); load L2fmri_LDTw; end

%% Figure 1
allclearL2
qw = 1:32; qn = 33:64; ql = 65:74;
RT = L2_str.RT;
nsubs = numel(L2_str.subjectid);
[idx, ROIname] = getvoxind(L2_str);

% Setting wrong responses to nan
correctresp(1:32,1:2:nsubs) = 55; correctresp(1:32,2:2:nsubs) = 41;
correctresp(33:64,1:2:nsubs) = 41; correctresp(33:64,2:2:nsubs) = 55;
CR = repmat([correctresp; zeros(10,nsubs)],[1 1 8]);
iscorrect = L2_str.responsekey == CR; % Finding the correct response key
RT(iscorrect == 0) = nan;
RT(RT == 0)= nan;  % excluding RTs with incorreact or no response

% Loading behaviour and fMRI dissimilarities
load dfmri_LDC; droi(isoutlier(droi,2)) = nan; % Removing outliers
dpsyw = L2_str.search.w;
dpsyn = L2_str.search.n;
dpsynw = L2_str.search.wnw;

% Loading semantic dissimilarity 
load figure5_semanticfeat; dsemfeat = pdist(semanticfeat,'correlation')';
% Code to generate semantic dissimilarities
% filename = "glove.42B.300d"; emb = readWordEmbedding(filename + '.txt');
% for i = 1:32, semanticfeat(i,:) = word2vec(emb,char(L2_str.stimid(i,:)+96))'; end

% Index of visual search pairs in fMRI dissimilarity values
wid = find(ismember(nchoosek(1:74,2),nchoosek(1:32,2),'rows'));
nid = find(ismember(nchoosek(1:74,2),nchoosek(1:32,2)+32,'rows'));
nwid = find(ismember(nchoosek(1:74,2),[1:32; 33:64]','rows'));
sid = find(ismember(nchoosek(1:74,2),nchoosek(1:64,2),'rows'));

% Averaging reaction time across runs and subjects
meanRTn = nanmean(nanmean(RT(qn,:,1:8),3),2); 
meanRTw = nanmean(nanmean(RT(qw,:,1:8),3),2);

medianRTn = nanmedian(nanmedian(RT(qn,:,1:8),3),2); % Median RT
medianRTw = nanmedian(nanmedian(RT(qw,:,1:8),3),2);

%% Analysis mentioned in main text 
% mean accuracy
Rmeanacc = mean(iscorrect,3);
% rmtrials = zeros(size(Rmeanacc)); rmtrials(Rmeanacc < .15) = 1; rmtrials = rmtrials';
avgacc = mean(mean(Rmeanacc)); stdacc = std(mean(Rmeanacc));

% Correlating word/nonword reaction time with word frequency/dissimilarity 
[r(1), p(1)] = corr(log10(L2_str.wordfreq),medianRTw,'rows','complete','type','spearman');
[r(2), p(2)] = corr(dpsynw,medianRTn,'rows','complete','type','spearman');

% Esitmating mean RT for transposed and substituted nonwords
for type = 1:2
    Sidx = vec(L2_str.stimtype((type-1)*4+1:type*4,:));
    avgnwordRT(type,:) = mean(meanRTn(Sidx));
    stdnwordRT(type,:) = std(meanRTn(Sidx));
end

% correlation between semantic and visual search space
[r_sempsy, p_sempsy] = corr(dpsyw,dsemfeat);


%% Figure 5C
for roi = 1:5
    [rw(roi,:),~] = bootstrp(1000,@corr,nanmean(droi(wid,:,roi),2),dpsyw);
    [rn(roi,:),~] = bootstrp(1000,@corr,nanmean(droi(nid,:,roi),2),dpsyn);
    [ra(roi,:),~] = bootstrp(1000,@corr,nanmean(droi([nid; wid;nwid],:,roi),2),[dpsyn;dpsyw;dpsynw]);
end

figure; barweb(mean(ra,2), std(ra,[],2)); xticklabels('All stimuli'); 
title('correlation between fMRI and behaviour dissimilarity'); ylabel('Correlation Coefficient')
legend('All stimuli'); legend(ROIname);


%% Figure 5D
for roi = 1:5
    [rs(roi,:),~] = bootstrp(1000,@corr,nanmedian(droi(wid,:,roi),2),dsemfeat);
end
figure; barweb(mean(rs,2),std(rs,[],2)); legend(ROIname); 
title('correlation between fMRI and Semantic dissimilarity'); ylabel('Correlation Coefficient')

%% Figure 5E and 5F
% Mean dissimilarities are better correlated with activity
cnt = 0; 
for roi = 1:5
    cnt = cnt + 1;
    for sub = 1:numel(L2_str.subjectid)
        betas = L2_str.mergedevtbeta{sub};
        if roi == 4; max_vox = 40; else, max_vox = Inf; end
        nvox = min(numel(idx{sub,roi}),max_vox);
        
        actw(roi,sub,:)  = nanmean(betas(idx{sub,roi}(1:nvox),qw));  % mean words responses
        actn(roi,sub,:)  = nanmean(betas(idx{sub,roi}(1:nvox),qn));  % mean nonword responses
        actl(roi,sub,:)  = nanmean(betas(idx{sub,roi}(1:nvox),ql));  % mean single letter responses
    end
    
    % Correlating word and nonword RT with mean activity values.
      [rRT(roi,1,:),~] = bootstrp(1000,@corr,squeeze(nanmean(actw(roi,:,:),2)),meanRTw);
      [rRT(roi,2,:),~] = bootstrp(1000,@corr,squeeze(nanmean(actn(roi,:,:),2)),meanRTn);
      [rRT_all(roi,1,:),~] = bootstrp(1000,@corr,[squeeze(nanmean(actw(roi,:,:),2));squeeze(nanmean(actn(roi,:,:),2))],[meanRTw;meanRTn]);
end

% Figure 5E
figure; corrplot([squeeze(mean(actw(4,:,:),2)) squeeze(mean(actn(4,:,:),2))],[meanRTw meanRTn]); hold on;
h(1) = plot(squeeze(mean(actw(4,:,:),2)),meanRTw,'or');
h(2) = plot(squeeze(mean(actn(4,:,:),2)),meanRTn,'sg');
legend(h,'word','nonword'); ylabel('mean RT (s)'); xlabel('VWFA activity')

% Figure 5F
figure; barweb(mean(rRT,3),std(rRT,[],3)); xticklabels(ROIname); ylabel('Correlation coefficient');
title('Correlation b/w RT and activity'); legend('Words','Nonwords')
