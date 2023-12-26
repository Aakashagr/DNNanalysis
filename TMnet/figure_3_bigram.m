allclear
load('L2_telmal_bigram.mat')

L2_str.RT(L2_str.RT < .3) = NaN;  % removing accidental key press.
mat = zeros(size(L2_str.RT)); mat(L2_str.RT>5) = 1; % identifying harder search pairs
temp = sum(sum(mat,2),3);
% figure; stem(temp) % identifying mistakes.
id = find(temp > 0 & temp < 2);

% removing higher RT outlier
for i = 1:length(id)
    [c, v] =ind2sub([size(L2_str.RT,2),2],find(mat(id(i),:,:)==1));
    L2_str.RT(id(i),c,v) = NaN;
end

images = L2_str.images;
image_pairs = L2_str.img_pairs;
RT = L2_str.RT;
part_id = L2_str.partsTD;

ismal = L2_str.subjinfo.ismalayalam;
RT_mal = RT(:,ismal == 1,:); 
RT_tel = RT(:,ismal == 0,:);
PC_mal = L2_str.PC(:,ismal == 1,:);
PC_tel = L2_str.PC(:,ismal == 0,:);

mRT_mal = nanmean(RT_mal,3);
mRT_tel = nanmean(RT_tel,3);

% Malayalam and telugu pairs
tel = 1:300; mal = 301:600;

% Average across subjects
RT_mal_mean = nanmean(mRT_mal,2);
RT_tel_mean = nanmean(mRT_tel,2);
RTmm = RT_mal_mean(mal);
RTmt = RT_mal_mean(tel);
RTtt = RT_tel_mean(tel);
RTtm = RT_tel_mean(mal);

% segregating high and low frequency bigrams
% Telugu
hfword_id_t = [3,8,11,12,14:18,23,24]; % high frequency bigram
nonword_id_t = setdiff(1:25,hfword_id_t); 

temp_hfword_t = hfword_id_t(nchoosek(1:numel(hfword_id_t),2));
temp_others_t = nonword_id_t(nchoosek(1:numel(nonword_id_t),2));

hfword_pairs_t = find(ismember(image_pairs,[temp_hfword_t; fliplr(temp_hfword_t)],'rows')==1);
lfword_pairs_t  = find(ismember(image_pairs,[temp_others_t; fliplr(temp_others_t)],'rows')==1);

% Malayalam
hfword_id_m = [27:29,33:34,36,37,39:43,46,48,49]; % high frequency bigram
lfword_id_m = setdiff(26:50,hfword_id_m); 

temp_hfword_m = hfword_id_m(nchoosek(1:numel(hfword_id_m),2));
temp_others_m = lfword_id_m(nchoosek(1:numel(lfword_id_m),2));

hfword_pairs_m = find(ismember(image_pairs,[temp_hfword_m; fliplr(temp_hfword_m)],'rows')==1);
lfword_pairs_m  = find(ismember(image_pairs,[temp_others_m; fliplr(temp_others_m)],'rows')==1);

% Baseline reaction time
for i = 1:numel(L2_str.BRT)
    brtmean(i,1) = mean(L2_str.BRT{i}(4:10));
end

% Correlation between readers and nonreaders
% cT = nancorrcoef(1./RTtt,1./RTmt);
% cM = nancorrcoef(1./RTmm,1./RTtm);

%% Panel B

figure; barweb([mean(RTtt), mean(RTmt); mean(RTtm), mean(RTmm); mean(brtmean(ismal==0)) mean(brtmean(ismal ==1))], ...
    [nansem(nanmean(mRT_tel(tel,:))'), nansem(nanmean(mRT_mal(tel,:))'); ...
    nansem(nanmean(mRT_tel(mal,:))'), nansem(nanmean(mRT_mal(mal,:))'); nansem(brtmean(ismal==0)) nansem(brtmean(ismal ==1))])
ylabel('Avg Search Time, Seconds');   colormap(get(gca,'ColorOrder'))
set(gca,'Xticklabel',{'Telugu Bigrams','Malayalam Bigrams','Baseline stimuli'});
legend({'Telugu readers','Malayalam readers'});

%% Baton model

[y_pred_tt,w_tt,~,mat] = Batonmodel(1./RTtt, 5, 2,1,0);
[y_pred_mt,w_mt] = Batonmodel(1./RTmt, 5, 2,1,0);
[y_pred_mm,w_mm] = Batonmodel(1./RTmm, 5, 2,1,0);
[y_pred_tm,w_tm] = Batonmodel(1./RTtm, 5, 2,1,0);

% sigmoid fit
[~,y_pred_new_tt] = fitsigmoid(y_pred_tt,1./RTtt);
[~,y_pred_new_mt] = fitsigmoid(y_pred_mt,1./RTmt);
[~,y_pred_new_tm] = fitsigmoid(y_pred_tm,1./RTtm);
[~,y_pred_new_mm] = fitsigmoid(y_pred_mm,1./RTmm);

p = statcomparemean(w_tt(1:10),w_mt(1:10));

% correlation between model terms for Telugu readers on Telugu bigrams
% [cT_CA, pT_CA] = nancorrcoef(w_tt(1:10),w_tt(11:20));
% [cT_CW, pT_CW] = nancorrcoef(w_tt(1:10),w_tt(21:30));

% correlating corresponding terms with single letter experiment
% load pair_RT_2baton
% [cSL_C,pSL_C] = nancorrcoef(w_tt(1:10),1./rt_tt);

%%  Panel D 

% Telugu on Telugu
figure; corrplot(y_pred_new_tt,1./RTtt,'Tel-Tel',1); hold all; 
h(1) = plot(y_pred_new_tt(lfword_pairs_t),1./RTtt(lfword_pairs_t),'rs'); 
h(2) = plot(y_pred_new_tt(hfword_pairs_t),1./RTtt(hfword_pairs_t),'go');%lsline
xlabel('Predicted dissimilarity'); ylabel('Observed dissimilarity');
legend(h,{'Low-freq bigram','High-freq bigram'});

% Malalayalam on Malayalam
% figure; corrplot(y_pred_new_mm,1./RTmm,'Mal-Mal',1); hold all; 
% h(1) = plot(y_pred_new_mm(lfword_pairs_m-300),1./RTmm(lfword_pairs_m-300),'rs'); 
% h(2) = plot(y_pred_new_mm(hfword_pairs_m-300),1./RTmm(hfword_pairs_m-300),'go');%lsline
% xlabel('Predicted dissimilarity'); ylabel('Observed dissimilarity');
% legend(h,{'Low-freq bigram','High-freq bigram'});


% Residual error for high and low frequency bigram pairs (Telugu bigrams)
% error_lf = y_pred_new_tt(lfword_pairs_t) - 1./RTtt(lfword_pairs_t);
% error_hf = y_pred_new_tt(hfword_pairs_t) - 1./RTtt(hfword_pairs_t);
% sqrt(nanmean((error_hf).^2));
% sqrt(nanmean((error_lf).^2))
% ranksum(error_lf,error_hf)

% Residual error for high and low frequency bigram pairs (Malayalam bigrams)
error_lf = y_pred_new_mm(lfword_pairs_m-300) - 1./RTmm(lfword_pairs_m-300);
error_hf = y_pred_new_mm(hfword_pairs_m-300) - 1./RTmm(hfword_pairs_m-300);
sqrt(nanmean((error_hf).^2));
sqrt(nanmean((error_lf).^2))
ranksum(error_lf,error_hf)
%% Panel X
% Calculating noise ceiling
% [~,cmm] = splithalfcorr(nanmean(RT_mal(mal,:,:),3)'); cmm = spearmanbrowncorrection(cmm,2);
% [~,ctt] = splithalfcorr(nanmean(RT_tel(tel,:,:),3)'); ctt = spearmanbrowncorrection(ctt,2);
% [~,cmt] = splithalfcorr(nanmean(RT_mal(tel,:,:),3)'); cmt = spearmanbrowncorrection(cmt,2);
% [~,ctm] = splithalfcorr(nanmean(RT_tel(mal,:,:),3)'); ctm = spearmanbrowncorrection(ctm,2);
% 
% r_tt = nancorrcoef(y_pred_new_tt,1./RTtt);
% r_tm = nancorrcoef(y_pred_new_tm,1./RTtm);
% r_mt = nancorrcoef(y_pred_new_mt,1./RTmt);
% r_mm = nancorrcoef(y_pred_new_mm,1./RTmm);
% 
% figure; bar([r_tt,r_mt r_mm,r_tm]); ylabel('Correlation Coefficient'); hold on
% set(gca,'XtickLabel',{'TT', 'MT', 'MM', 'TM'}); title('Model Prediction')
% shadedErrorBar([0.8,1.2],[mean(ctt) mean(ctt)], [std(ctt) std(ctt)])
% shadedErrorBar([1.8,2.2],[mean(cmt) mean(cmt)], [std(cmt) std(cmt)])
% shadedErrorBar([2.8,3.2],[mean(cmm) mean(cmm)], [std(cmm) std(cmm)])
% shadedErrorBar([3.8,4.2],[mean(ctm) mean(ctm)], [std(ctm) std(ctm)])


%% comparing RT difference
% figure; subplot(121); corrplot(1./y_pred_new_tt - 1./y_pred_new_mt, RTtt - RTmt)
%  subplot(122); corrplot(1./y_pred_new_mm - 1./y_pred_new_tm, RTmm - RTtm)

%% Panel E and F
xx = reshape(w_tt(1:30),[10,3]);
figure;barweb([mean(reshape(w_tt(1:30),[10,3]));mean(reshape(w_mt(1:30),[10,3]))]',[nansem(reshape(w_tt(1:30),[10,3])); nansem(reshape(w_mt(1:30),[10,3]))]');
legend('Native readers','Non readers'); title('Telugu bigram');
xlabel('Parameters'); ylabel('Magnitude');
set(gca,'XtickLabel',{'Corresponding part terms','Across part terms','Within part terms'});

figure; barweb([mean(reshape(w_mm(1:30),[10,3]));mean(reshape(w_tm(1:30),[10,3]))]',[nansem(reshape(w_mm(1:30),[10,3])); nansem(reshape(w_tm(1:30),[10,3]))]');
legend('Native readers','Non readers'); title('Malayalam bigram');
xlabel('Parameters'); ylabel('Magnitude');
set(gca,'XtickLabel',{'Corresponding part terms','Across part terms','Within part terms'});

% STATS
xt = reshape(w_tt(1:30),[10,3]); 
xm = reshape(w_mt(1:30),[10,3]);
yt = reshape(w_tm(1:30),[10,3]);
ym = reshape(w_mm(1:30),[10,3]);


p = statcomparemean(xt(:,1), xm(:,1));

%% Panel G
% AB-BA pairs
pairs = [2 6; 3 11; 4 16; 5 21; 8 12; 9 17; 10 22; 14 18; 15 23; 20 24]; ids1 = find(ismember(nchoosek(1:25,2),pairs,'rows'));
pairs = nchoosek([1 7 13 19 25],2); ids = find(ismember(nchoosek(1:25,2),pairs,'rows')); 

figure; barweb([[mean([RTtt(ids1); RTmm(ids1)]); mean([RTtm(ids1); RTmt(ids1)])]'; [mean([RTtt(ids); RTmm(ids)]); mean([RTtm(ids); RTmt(ids)])]'], ...
    [[nansem([nanmean(mRT_tel(ids1,:))';nanmean(mRT_mal(ids1+300,:))']); nansem([nanmean(mRT_tel(ids1+300,:))'; nanmean(mRT_mal(ids1,:))'])]'; ...
    [nansem([nanmean(mRT_tel(ids,:))';nanmean(mRT_mal(ids+300,:))']); nansem([nanmean(mRT_tel(ids+300,:))'; nanmean(mRT_mal(ids,:))'])]'])
ylabel('Avg Search Time, Seconds');   colormap(get(gca,'ColorOrder'))
set(gca,'Xticklabel',{'AB vs BA','AA vs BB'});
legend({'readers','non-readers'});

% mean RT for AA BB types searches separately for each group
% [mean(RTtt(ids)) mean(RTmt(ids)) mean(RTmm(ids)) mean(RTtm(ids))];
% ranksum(RTtt(ids),RTmt(ids));
% ranksum(RTmm(ids),RTtm(ids))

%% Stats
% Telugu & Malayalam letters separately ters
cnt = 0;
for imgpairs = 1:300
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
            
pt = anovan(TRT,{grp,IMGpairs},'model','full'); % p < 10-3
pm = anovan(MRT,{grp,IMGpairs},'model','full'); % p < 10-5

%% comparing RT and 1/RT models
% for Telugu letters
b_tt = regress(1./RTtt,mat); y_pred_tt = mat*b_tt;
b_mt = regress(1./RTmt,mat); y_pred_mt = mat*b_mt;
% w_tm = regress(1./RTtm,mat); y_pred_tm = mat*w_tm; 
% w_mm = regress(1./RTmm,mat); y_pred_mm = mat*w_mm;

r_tt = nancorrcoef(y_pred_tt,1./RTtt);
r_mt = nancorrcoef(y_pred_mt,1./RTmt);
% r_tm = nancorrcoef(y_pred_tm,1./RTtm);
% r_mm = nancorrcoef(y_pred_mm,1./RTmm);
% [am,as] = aicc(1./y_pred_tt,RTtt,31)

%% On separate parts
cidx = [1:10 31]; b = regress(1./RTtt,mat(:,cidx)); ypredtt_C = mat(:,cidx)*b;
cidx = [11:20 31]; b = regress(1./RTtt,mat(:,cidx)); ypredtt_A = mat(:,cidx)*b;
cidx = [21:30 31]; b = regress(1./RTtt,mat(:,cidx)); ypredtt_W = mat(:,cidx)*b;
cidx = [1:20 31]; b = regress(1./RTtt,mat(:,cidx)); ypredtt_CA = mat(:,cidx)*b;
cidx = [1:10 21:31]; b = regress(1./RTtt,mat(:,cidx)); ypredtt_CW = mat(:,cidx)*b;
cidx = [11:31]; b = regress(1./RTtt,mat(:,cidx)); ypredtt_AW = mat(:,cidx)*b;

cidx = [1:10 31]; b = regress(1./RTmt,mat(:,cidx)); ypredmt_C = mat(:,cidx)*b;
cidx = [11:20 31]; b = regress(1./RTmt,mat(:,cidx)); ypredmt_A = mat(:,cidx)*b;
cidx = [21:30 31]; b = regress(1./RTmt,mat(:,cidx)); ypredmt_W = mat(:,cidx)*b;
cidx = [1:20 31]; b = regress(1./RTmt,mat(:,cidx)); ypredmt_CA = mat(:,cidx)*b;
cidx = [1:10 21:31]; b = regress(1./RTmt,mat(:,cidx)); ypredmt_CW = mat(:,cidx)*b;
cidx = [11:31]; b = regress(1./RTmt,mat(:,cidx)); ypredmt_AW = mat(:,cidx)*b;

predtt = [ypredtt_C ypredtt_A ypredtt_W];
predmt = [ypredmt_C ypredmt_A ypredmt_W];

[r, p] = partialcorri(1./RTmt,predmt);
[am,as] = aicc(ypredmt_W,1./RTmt,11);

[rc, pc] = partialcorri(1./RTmt,[ypredmt_AW ypredmt_C])
[am,as] = aicc(ypredmt_CW,1./RTmt,21)

%% Understanding within part-terms
% load('tel_25bigram.mat')
% xx = tel_bigram([2 3 4 5 8 9 10 14 15 20]);
% figure; subplot(121);  corrplot(xx, (w_tt(1:10)-abs(w_tt(21:30))));  xlabel('Bigram frequency'); ylabel('corr - abs(within) terms')
% subplot(122); corrplot(w_tt(1:10),abs(w_tt(21:30)),[],1); xlabel('Corresponding terms'); ylabel('Absolute Within terms')
% 
% load('mal_25bigram.mat')
% xx = mal_bigram([2 3 4 5 8 9 10 14 15 20]);
% figure; subplot(121);  corrplot(xx, (w_mm(1:10)-abs(w_mm(21:30))));  xlabel('Bigram frequency'); ylabel('corr - abs(within) terms')
% subplot(122); corrplot(w_mm(1:10),abs(w_mm(21:30)),[],1); xlabel('Corresponding terms'); ylabel('Absolute Within terms')
