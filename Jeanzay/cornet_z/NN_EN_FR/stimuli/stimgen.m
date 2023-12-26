allclear
load EN_FR_Hierarchy_stimlist.mat

% Identifying stimulus category
for i = 1:211
    stimcat(i,1) = stimlist(i).cat; 
    stimtype{i,1} = stimlist(i).code; 
end

% Saving image
for i = 1:14
    dirpath = ['HierStim/train/',num2str(i+1,'%02d'),'_/' ];
    mkdir(dirpath)
    idx = find(stimcat == i);
    cnt = 0;
    for catid = 1:numel(idx)
        strlist = stimlist(idx(catid)).content(:,1);
        for nstr = 1:numel(strlist)
            img = ones(500,500,3)*255;
            img = insertText(img,[250,250],strlist{nstr},'Font','Arial','Fontsize',60,'AnchorPoint','center');
            imwrite(img,[dirpath, num2str(cnt,'%02d'),'.jpg'])
            cnt = cnt+1;
        end
    end
end