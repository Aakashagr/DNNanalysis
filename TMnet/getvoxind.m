%  idx = getvoxind(L2_str)
% Get voxels indices common between functional and anatomical ROIs

function [idx, ROInames, aROIs] = getvoxind(L2_str,isanat)
if (~exist('isanat')); isanat = 0; end
aROIs = {'Visual_hOc1_L'; 'Visual_hOc1_R'; 'Visual_hOc2_L'; 'Visual_hOc2_R';  'Visual_hOc3d_L'; 'Visual_hOc3d_R'; 'Visual_hOc3v_L'; 'Visual_hOc3v_R';...
    'Visual_hOc4d_L'; 'Visual_hOc4d_R'; 'Visual_hOc4v_L'; 'Visual_hOc4v_R'; 'MOG_L'; 'MOG_R'; 'ITG_L'; 'ITG_R'; ...
    'IOG_L'; 'IOG_R'; 'STG_L'; 'STG_R'; 'MTG_L'; 'MTG_R'; 'SMG_L'; 'SMG_R'; 'PCG_L'; 'PCG_R';...
    'Visual_FG1_L';  'Visual_FG1_R'; 'Visual_FG2_L'; 'Visual_FG2_R'; 'Visual_FG3_L'; 'Visual_FG3_R'; 'Visual_FG4_L'; 'Visual_FG4_R'};

fROIs = {'EVC','LOC','VWFA','WSCR'};

for sub = 1:numel(L2_str.ismal)
    for roi = 1:5
        fidx = []; aidx = [];
        % Identifying anatomical and functional indices.
        switch roi
            case 1
                fidx = L2_str.fROI.(fROIs{1}).ids{sub};
                tvalues = L2_str.fROI.(fROIs{1}).tvalues{sub};
                for aR = 1:8, aidx = [aidx; L2_str.aROI.(aROIs{aR}){sub}];end
                ROInames{1} = 'V1-V3';
            case 2
                fidx = union(L2_str.fROI.(fROIs{1}).ids{sub},L2_str.fROI.(fROIs{2}).ids{sub});
                tvalues = union(L2_str.fROI.(fROIs{1}).tvalues{sub},L2_str.fROI.(fROIs{2}).tvalues{sub});
                for aR = 9:12, aidx = [aidx; L2_str.aROI.(aROIs{aR}){sub}];end
                ROInames{2} = 'V4';
            case 3
                fidx = L2_str.fROI.(fROIs{2}).ids{sub};
                tvalues = L2_str.fROI.(fROIs{2}).tvalues{sub};
                for aR = 13:18, aidx = [aidx; L2_str.aROI.(aROIs{aR}){sub}];end
                ROInames{3} = 'LOC';
            case 4
                fidx = L2_str.fROI.(fROIs{3}).ids{sub};
                tvalues = L2_str.fROI.(fROIs{3}).tvalues{sub};
%                 for aR = 27:34, aidx = [aidx; L2_str.aROI.(aROIs{aR}){sub}];end
                ROInames{4} = 'VWFA';
            case 5
                fidx = L2_str.fROI.(fROIs{4}).ids{sub};
                tvalues = L2_str.fROI.(fROIs{4}).tvalues{sub};
                for aR = 19:22, aidx = [aidx; L2_str.aROI.(aROIs{aR}){sub}];end
                ROInames{5} = 'TG';
        end
        
        % Identifying common voxels and sorting them based on tvalues
        if (isempty(aidx)), aidx = fidx; end        
        if isanat
            idx{sub,roi} = aidx;
        else
            comvox = intersect(fidx, aidx);
            [~,I] = sort(tvalues(ismember(fidx, comvox)),'descend');
            idx{sub,roi} = comvox(I);
        end
    end
    % Selecting unique voxels
    idx{sub,3} = setdiff(idx{sub,3},[idx{sub,1}; idx{sub,2}; idx{sub,4}; idx{sub,5}]);
end