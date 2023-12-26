% clear; clc
% Load data structure
% load L2fmri_READINGw
% age = std([24 24 25 22 20 24 25 24 27 29 25 24 22 23 26 25 34 26 23 25 26 25 26 26 32 32 23 30 24 21 21 30 23 23 19])
ismal = L2_str.ismal; % Is subject Malayalam: 1 (for Malayalam) and 0 (for Telugu)
qm = 1:34; qt = 35:68; % Index of Malayalam and Telugu stimuli
qs = 1:10; qd = 11:34; % Index of single and double letter stimuli
[ids, ROIname] = getvoxind(L2_str); % Extracting voxel IDs of various Region of Interest (ROI)-interesection of functional and anatomical regions
images = L2_str.evtimages([qm, qt]);
%%
distmsr = 'spearman'; % 1- r (correlation coefficient) is used as a dissimilarity measure.

% Extracting dissimilarity for each subject and each ROI
Tdis = []; Mdis = []; 
for roi = 1:5
    for sub = 1:numel(L2_str.mergedevtbeta)
        betas = L2_str.mergedevtbeta{sub};                      % Extracting Beta (or regression weights) for each subject
        if roi == 4; max_vox = 20; elseif roi == 3, max_vox = 200; else, max_vox = Inf; end     % Restricting VWFA definitions upto top 75 voxels
%         if roi == 4; max_vox = 15; elseif roi == 3, max_vox = 100; else, max_vox = Inf; end     % Restricting VWFA definitions upto top 75 voxels
        nvox = min(numel(ids{sub,roi}),max_vox);                % Total number of voxels considered in the analysis
        
        % Calculating pair-wise dissimilarities for double letter stimuli
        xx = betas(ids{sub,roi}(1:nvox),qm(qd))'; xx(:,isnan(mean(xx))) = []; Mdis(roi,sub,:) = pdist(xx,distmsr);
        xx = betas(ids{sub,roi}(1:nvox),qt(qd))'; xx(:,isnan(mean(xx))) = []; Tdis(roi,sub,:) = pdist(xx,distmsr);
    end
end


save dfmri Tdis Mdis ismal images 