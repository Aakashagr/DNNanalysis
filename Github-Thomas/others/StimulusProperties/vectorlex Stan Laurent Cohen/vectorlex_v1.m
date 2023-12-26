addpath Q:\imagerie\generic\matlab\myprograms
%% path to gettextdatafromexcel.m

if ~exist('manuall.mat')
    dico = 'C:\Users\Stanislas Dehaene\hubic\education\lecture\lilianeSprengerCharolles\manuall-10000mots-18-09-2010.xls';
    ortho = gettextdatafromexcel(dico,'ORTHO');
    flexions = gettextdatafromexcel(dico,'FLEXIONS');
    categ = gettextdatafromexcel(dico,'CATEG');
    freq = getnumdatafromexcel(dico,'FREQ');
    psyll = gettextdatafromexcel(dico,'PSYLL');
    gpmatch = gettextdatafromexcel(dico,'GPMATCH');
    nbgraph = getnumdatafromexcel(dico,'NBGRAPH');
    nblet = getnumdatafromexcel(dico,'NBLET');
    nbphon = getnumdatafromexcel(dico,'NBPHON');
    nbsyll = getnumdatafromexcel(dico,'NBSYLL');
    nwords = length(ortho);
    save 'manuall.mat' ortho flexions freq flex categ psyll gpmatch nbgraph nblet nbphon nbsyll nwords
else
    load manuall.mat
end

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

nletters = 5;

%%% reduce to just the list of words with n letters
listw = find(nblet == nletters);
ortho = ortho(listw);
freq = freq(listw);
categ = categ(listw);
psyll = psyll(listw);
gpmatch = gpmatch(listw);
nbgraph = nbgraph(listw);
nblet = nblet(listw);
nbphon = nbphon(listw);
nbsyll = nbsyll(listw);
flexions = flexions(listw);

orthou = unique(ortho); %% 1 exemplar of each word
nwords = length(orthou);

%%% create the letter vector for each word
vectorlex = zeros(nwords,26*nletters);
for iw = 1:nwords %%% scan all words
    thisword = orthou{iw};
    neighborspread = 0.0; %% spread to nearby letters
    for il = 1:length(thisword)
        thisletter = thisword(il);
        vectorlex(iw,26*(il-1)+(cast(thisletter,'int16')-96)) = 1;
        %%% smear the activation also to neighboring letter positions:
        if il>1
            lpos = 26*((il-1)-1)+(cast(thisletter,'int16')-96);
            vectorlex(iw,lpos) = vectorlex(iw,lpos) + neighborspread;        
        end
        if il<nletters
            lpos = 26*((il-1)+1)+(cast(thisletter,'int16')-96);
            vectorlex(iw,lpos) = vectorlex(iw,lpos) + neighborspread;        
        end
    end,
    %% normalize so that each word vector lies on the unit sphere
    vectorlex(iw,:) = vectorlex(iw,:)/norm(vectorlex(iw,:));
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

meanvectorlex = mean(vectorlex,1);
figure(2);
title('Mean word vector');
bar(meanvectorlex);

for iw = 1:nwords
    vectorlex(iw,:) = vectorlex(iw,:) - meanvectorlex;
end

[coeff,score,latent,tsquared,explained,mu] = pca(vectorlex);

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

ncomponents = 40;
%%%% make a figure of the first n components
figure(10);
clf;
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

%%% generate random "words" as points in this hyperspace, and check what
% they look like
ngenerated = 50;
for iw = 1:ngenerated
    coords = randn(1,ncomponents);
    coords = coords / norm(coords);
    randomword = (coords * coeff(:,1:ncomponents)') + meanvectorlex;
    for il= 1:nletters
        [m,i] = max(randomword((1:26)+26*(il-1)));
        random_orthou(il) = cast(i+96,'char');
    end
    disp(sprintf('%s',random_orthou));
end
