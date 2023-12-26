%% Figure 2: Average RT and dissimilarity correlation plot.
allclear
load L2_letters

% removing outliers
L2_str.RT(L2_str.RT < .3) = NaN;  % removing accidental key press.
mat = zeros(size(L2_str.RT)); mat(L2_str.RT>5) = 1; % identifying harder search pairs
temp = sum(sum(mat,2),3);
% figure; stem(temp) % identifying mistakes.
id = find(temp > 0 & temp < 7);

% removing higher RT outlier
for i = 1:length(id)
    [c, v] =ind2sub([size(L2_str.RT,2),2],find(mat(id(i),:,:)==1));
    L2_str.RT(id(i),c,v) = NaN;
end

% % subjects to remove
L2_str.RT(:,[29 28 12],:) = [];
L2_str.subjinfo.ismalayalam([29 28 12]) = [];
L2_str.fluencytest.reading_time([29 28 12]) = [];

images = L2_str.images;
image_pairs = L2_str.img_pairs(1:1260,:);
RT = L2_str.RT; PC = L2_str.PC;
ismal = L2_str.subjinfo.ismalayalam;
% Search time
RT_mal = RT(:,ismal == 1,:); RT_tel = RT(:,ismal == 0,:);
mRT_mal = nanmean(RT_mal,3); mRT_tel = nanmean(RT_tel,3);

% Average across subjects and repeats
RT_mal_mean = nanmean(mRT_mal,2);
RT_tel_mean = nanmean(mRT_tel,2);

% Malayalam and telugu pairs
mal = 1:630; tel = 631:1260;

% Calculating the mean percent correct across subjects and trials
% PC_mm = mean(mean(PC(mal,ismal == 1),2));  PC_tt = mean(mean(PPC(tel,ismal == 0),2));
% PC_mt = mean(mean(PPC(tel,ismal == 1),2)); PC_tm = mean(mean(PPC(mal,ismal == 0),2));

% Baseline reaction time
for i = 1:numel(L2_str.BRT)
    brtmean(i,1) = mean(L2_str.BRT{i}(4:10));
end

% Calculating the dissimilarity between the odd and even numbered subjects
% ctt = nancorrcoef(1./nanmean(mRT_tel(tel,1:2:end),2),1./nanmean(mRT_tel(tel,2:2:end),2));
% cmt = nancorrcoef(1./nanmean(mRT_mal(tel,1:2:end),2),1./nanmean(mRT_mal(tel,2:2:end),2));
% cmm = nancorrcoef(1./nanmean(mRT_mal(mal,1:2:end),2),1./nanmean(mRT_mal(mal,2:2:end),2));
% ctm = nancorrcoef(1./nanmean(mRT_tel(mal,1:2:end),2),1./nanmean(mRT_tel(mal,2:2:end),2));

% Change in dissimilarity
% c_disT = nancorrcoef(1./nanmean(mRT_mal(tel,:),2), 1./nanmean(mRT_tel(tel,:),2) - 1./nanmean(mRT_mal(tel,:),2));
% c_disM = nancorrcoef(1./nanmean(mRT_tel(mal,:),2), 1./nanmean(mRT_mal(mal,:),2) - 1./nanmean(mRT_tel(mal,:),2));
%%  Panel B. bar plot summarizing avgRT for each group & each language
data = [nanmean(RT_tel_mean(tel)) nanmean(RT_mal_mean(tel)); nanmean(RT_tel_mean(mal)) nanmean(RT_mal_mean(mal)); mean(brtmean(ismal==0)) mean(brtmean(ismal ==1))];
figure; datae= [nansem(nanmean(mRT_tel(tel,:))') nansem(nanmean(mRT_mal(tel,:))') ;nansem(nanmean(mRT_tel(mal,:))') nansem(nanmean(mRT_mal(mal,:))'); nansem(brtmean(ismal==0)) nansem(brtmean(ismal ==1))];

barweb(data,datae,[]);
set(gca,'XTickLabel',{'Telugu letters','Malayalam letters','Baseline stimuli'});
legend('Telugu readers','Malayalam readers');
ylabel('Avg Search time, seconds');
%% Panel C and D. Scatter plot of searches 1/RT averaged across subjects

figure; corrplot(1./RT_mal_mean(tel),1./RT_tel_mean(tel),'Telugu letters');unityslope;
ylabel('Telugu readers 1/RT (1/s)');xlabel('Malayalam readers 1/RT (1/s)')

% on Malayalam language
figure; corrplot(1./RT_tel_mean(mal),1./RT_mal_mean(mal),'Malayalam letters');unityslope;
ylabel('Malayalam readers 1/RT (1/s)');xlabel('Telugu readers 1/RT (1/s)')

%% bar plot of correlation
% bar([nancorrcoef(1./RT_tel_mean(mal),1./RT_mal_mean(mal)); nancorrcoef(1./RT_tel_mean(mal),1./RT_mal_mean(mal))])

%% Stats
% Telugu & Malayalam letters separately ters
cnt = 0;
for imgpairs = 1:630
    for sub = 1:size(RT,2)
        for reps = 1:size(RT,3)
            cnt = cnt + 1;
            TRT(cnt) = RT(tel(imgpairs),sub,reps);
            MRT(cnt) = RT(mal(imgpairs),sub,reps);
            grp(cnt) = ismal(sub);
            IMGpairs(cnt) = imgpairs;
        end
    end
end

% pt = anovan(TRT,{grp,IMGpairs},'model','full'); % p < 10-40
% pm = anovan(MRT,{grp,IMGpairs},'model','full'); % p < 10-102

