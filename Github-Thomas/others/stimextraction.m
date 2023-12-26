% Extracting hierarchical stimuli 
allclear
load HierarchyExp_stimlist_145.mat
for i = 1:numel(stimlist); cat(i) = stimlist(i).cat; end

[~,b] = unique(cat); cnt = 1;
for id = 1:numel(b)-1
    stim{cnt} = stimlist(b(id)).content{1}; cnt = cnt+1;
    stim{cnt} = stimlist(b(id)+1).content{1}; cnt = cnt+1;
end

%% Bilingual stimuli - Minye
allclear
load('Bilingual_Hierarchy_stimlist.mat')
for i = 1:numel(stimlist); cat(i) = stimlist(i).cat; end
[~,b] = unique(cat); cnt = 1;
for id = 1:numel(b)-1
    stim{cnt} = stimlist(b(id)).content{1}; cnt = cnt+1;
end

