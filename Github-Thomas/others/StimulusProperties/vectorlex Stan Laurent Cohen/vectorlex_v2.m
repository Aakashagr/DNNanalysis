addpath Q:\imagerie\generic\matlab\myprograms
%% path to gettextdatafromexcel.m

if ~exist('lexique_fr.mat')
    dico = 'C:\Users\Stanislas Dehaene\hubic\database\LEXIQUE\Lexique3.orthofreq.txt';
    fileID = fopen(dico);
    C = textscan(fileID,'%s\t%f%*[^\n]');
    fclose(fileID);
    ortho=C{1};
    freq=C{2};
    nblet = cellfun(@length,ortho);
    save 'lexique_fr.mat' ortho freq nblet
end
if ~exist('lexique_en.mat')
    dico = 'C:\Users\Stanislas Dehaene\hubic\database\DicoEnglish\dico.txt';
    fileID = fopen(dico);
    C = textscan(fileID,'%s\t');
    fclose(fileID);
    ortho=C{1};
    nblet = cellfun(@length,ortho);
    freq = 10 * ones(size(nblet)); %% no frequency available in this dictionnary
    save 'lexique_en.mat' ortho freq nblet
end

load lexique_en.mat

ortho = strrep(ortho,'à','a');
ortho = strrep(ortho,'â','a');
ortho = strrep(ortho,'ä','a');
ortho = strrep(ortho,'é','e');
ortho = strrep(ortho,'è','e');
ortho = strrep(ortho,'ê','e');
ortho = strrep(ortho,'ë','e');
ortho = strrep(ortho,'î','i');
ortho = strrep(ortho,'ï','i');
ortho = strrep(ortho,'ô','o');
ortho = strrep(ortho,'û','u');
ortho = strrep(ortho,'ù','u');
ortho = strrep(ortho,'ç','c');

%%% check for other bad characters
badchars = zeros(length(ortho),1);
for iw=1:length(ortho)
    if strfind(ortho{iw},'_')
        badchars(iw)=1;
    end
    if strfind(ortho{iw},'-')
        badchars(iw)=1;
    end
    if strfind(ortho{iw},'''')
        badchars(iw)=1;
    end
end

nletters = 4;

%%% reduce to just the list of words with n letters
listw = find((nblet == nletters).*(freq>=10).*(badchars==0));
ortho = ortho(listw);
nblet = nblet(listw);
freq = freq(listw);

orthou = unique(ortho); %% 1 exemplar of each word
nwords = length(orthou);

%%% create the letter vector for each word
vectorlex = zeros(nwords,26*nletters);
neighborspread = 0.1; %% spread to nearby letters
for iw = 1:nwords %%% scan all words
    vectorlex(iw,:)=makevector(orthou{iw},neighborspread);
end

figure(1); 
title('Original word codes');
imagesc(vectorlex);
xlabel('letter codes');
ylabel('words');

%% show all the word codes
% figure(10);
% for iw=1:nwords
%     clf;
%     vec2word_v1(vectorlex(iw,:));
%     disp(orthou{iw});
%     pause;
% end

centered = true; %%% better to center the data
meanvectorlex = mean(vectorlex,1);
if ~centered %% no mean subtraction
   meanvectorlex = zeros(size(meanvectorlex)); 
else
    figure(2);
    title('Mean word vector');
    bar(meanvectorlex);
    vec2word(meanvectorlex);
end

for iw = 1:nwords
    vectorlex(iw,:) = vectorlex(iw,:) - meanvectorlex;
end

[coeff,score,latent,tsquared,explained,mu] = pca(vectorlex,'centered',centered);

%%% Various checks to see that I understand the matrix structures
word1 = vectorlex(1,:);
%%% here is the formula to recover the scores (= loadings) for any word:
score1 = word1 * coeff ; 
%figure(99);
%plot(score1,'b');plot(score(1,:)); %%% these are identical !
%%% and to recover the Hotelling t square:
varscores = var(score);  %%% same as "explained" variance
df = length(score);  %%% that doesn't work
df = 94; %%% see below, essential to take into account the real degrees of freedom
%%% to get this, open pca.m and put a breakpoint at line 507
%%%         warning(message('stats:pca:ColRankDefX', q)); 

%% Only this formula allows to compute a distance for every word, including new ones!
for iw=1:nwords
    tsquare(iw) = sum((score(iw,1:df).^2)./varscores(1:df));
    %%% according to https://learnche.org/pid/latent-variable-modelling/principal-component-analysis/hotellings-t2-statistic
end
tsquarem = mahal(score(:,1:df),score(:,1:df)); %%% according to matlab doc)
figure(99);
clf; hold on;
plot(tsquared,tsquare,'.b');
plot(tsquared,tsquarem,'.r');
plot(tsquare,tsquarem,'.g');
%%% none of these measures coincide!!! I dont understand what tsquared the
%%% PCA computes...
%%% The reason is in the PCA function tsquared = localTSquared(score, latent,DOF, p)
% Subfunction to calulate the Hotelling's T-squared statistic. It is the
% sum of squares of the standardized scores, i.e., Mahalanobis distances.
% When X appears to have column rank < r, ignore components that are
% orthogonal to the data.
% when we set the degrees of freedom to the proper number (here 94) then
% all is fine, the formulas are correct.


figure(3);
title('Principal components, in order of importance');
imagesc(coeff);
xlabel('Component #');
ylabel('Values over letter codes');

figure(4);
title('Variance explained as a function of the number of components');
plot(cumsum(explained));

figure(5);
title('Projections of the words onto components');
imagesc(score);
xlabel('Loading on component #');
ylabel('Word');

%%%% make a figure of the first n components
figure(10);
clf;
ncomponents = 40;
for i =1:ncomponents;
    subplot(2,round(ncomponents/2),i)
    vec2word_v1(coeff(:,i));
    title(sprintf('%i',i));
end

%%%% reconstruct using a limited number of Principal Components
reconstructed = (score(:,1:ncomponents) * coeff(:,1:ncomponents)') + repmat(meanvectorlex,nwords,1);

figure(6);
title('Reconstructions of the original words');
imagesc(reconstructed);

%%%% recode the reconstruction as a word
for iw = 1:nwords
    for il= 1:nletters
        [m,i] = max(reconstructed(iw,(1:26)+26*(il-1)));
        recon_orthou{iw}(il) = cast(i+96,'char');
    end
    disp(sprintf('%s %s',orthou{iw},recon_orthou{iw}));
end

%%%% sort by Hotelling t squared
%%% attention, using variable tsquare here would mean that we are using only the
%%% first n components, not the entire set.
disp('ORIGINAL LEXICON sorted by Hotelling t squared');
[b,i]=sort(tsquare);
for iw=1:nwords
    disp(sprintf('%s %5.1f',orthou{i(iw)},tsquare(i(iw))));
end

%%%% load the original Grainger stimuli and see if we can account for them
load('GraingerStimuli.mat');
ngrainger = length(word);
for iw=1:ngrainger
   v=makevector(word{iw},neighborspread);
   tsquaregrainger(iw) = sum((score(iw,1:df).^2)./varscores(1:df));  
end
[b,i]=sort(tsquaregrainger);
for iw=1:ngrainger
    disp(sprintf('%s %5.1f',word{i(iw)},tsquaregrainger(i(iw))));
end
mean(tsquaregrainger(label_word==0))
mean(tsquaregrainger(label_word==1))

%%% generate random "words" as points in this hyperspace, and check what
% they look like
ngenerated = 50;
for iw = 1:ngenerated
    coords = randn(1,ncomponents);
    coords = coords / norm(coords);
    randomword = (coords * coeff(:,1:ncomponents)');
    randomword = randomword ./ norm(randomword);
    randomword = randomword + meanvectorlex;
    for il= 1:nletters
        [m,i] = max(randomword((1:26)+26*(il-1)));
        random_orthou(il) = cast(i+96,'char');
    end
    disp(sprintf('%s',random_orthou));
    figure(11);
    clf;
    vec2word_v1(randomword);
    pause;
end

