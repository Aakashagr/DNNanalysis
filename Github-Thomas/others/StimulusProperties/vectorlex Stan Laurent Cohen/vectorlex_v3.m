%%% this version of vectorlex has words of variable length 
%% corfou, en vue de Igoumenitza 9:44

addpath Q:\imagerie\generic\matlab\myprograms
%% path to gettextdatafromexcel.m

neighborspread = 0.1; %% spread to nearby letters

if ~exist('lexique.mat')
    dico = 'C:\Users\Stanislas Dehaene\hubic\database\LEXIQUE\Lexique3.orthofreq.txt';
    fileID = fopen(dico);
    C = textscan(fileID,'%s\t%f%*[^\n]');
    fclose(fileID);
    ortho=C{1};
    freq=C{2};
    nblet = cellfun(@length,ortho);
    save 'lexique.mat' ortho freq nblet
else
    load lexique.mat
end

ortho = strrep(ortho,'�','a');
ortho = strrep(ortho,'�','a');
ortho = strrep(ortho,'�','a');
ortho = strrep(ortho,'�','e');
ortho = strrep(ortho,'�','e');
ortho = strrep(ortho,'�','e');
ortho = strrep(ortho,'�','e');
ortho = strrep(ortho,'�','i');
ortho = strrep(ortho,'�','i');
ortho = strrep(ortho,'�','o');
ortho = strrep(ortho,'�','u');
ortho = strrep(ortho,'�','u');
ortho = strrep(ortho,'�','u');
ortho = strrep(ortho,'�','c');

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

minletters = 3;
maxletters = 8;
nletters = maxletters;

%%% reduce to just the list of words with n letters
listw = find((nblet >= minletters).*(nblet <= maxletters).*(freq>=10).*(badchars==0));
ortho = lower(ortho(listw));
nblet = nblet(listw);
freq = freq(listw);

orthou = unique(ortho); %% 1 exemplar of each word
nwords = length(orthou);

%%% create the letter vector for each word
vectorlex = zeros(nwords,26*nletters);
for iw = 1:nwords %%% scan all words
    thisword = orthou{iw};
    for il = 1:min(floor(nletters/2),length(thisword))   %% from left to right
        thisletter = thisword(il);
        numletter = (cast(thisletter,'int16')-96);
        lpos = 26*(il-1)+ numletter;
        vectorlex(iw,lpos) = vectorlex(iw,lpos) + 1;
        %%% smear the activation also to neighboring letter positions:
        if il>1
            vectorlex(iw,lpos-26) = vectorlex(iw,lpos-26) + neighborspread;        
        end
        if il<nletters
            vectorlex(iw,lpos+26) = vectorlex(iw,lpos+26) + neighborspread;        
        end
    end
    for il = 1:min(ceil(nletters/2),length(thisword))   %% from right to left
        thisletter = thisword(length(thisword)+1-il);
        numletter = (cast(thisletter,'int16')-96);
        lpos = 26*(nletters-il) + numletter;
        vectorlex(iw,lpos) = vectorlex(iw,lpos) + 1;
        %%% smear the activation also to neighboring letter positions:
        if il>1
            vectorlex(iw,lpos+26) = vectorlex(iw,lpos+26) + neighborspread;        
        end
        if il<nletters
            vectorlex(iw,lpos-26) = vectorlex(iw,lpos-26) + neighborspread;        
        end
    end
    %% normalize so that each word vector lies on the unit sphere
    vectorlex(iw,:) = vectorlex(iw,:)/norm(vectorlex(iw,:));
end

figure(1); 
title('Original word codes');
imagesc(vectorlex);
xlabel('letter codes');
ylabel('words');

% show all the word codes
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

ncomponents = 30;
%%%% make a figure of the first n components
figure(10);
clf;
for i =1:ncomponents; %% above 30 the figure gets crowded!
    subplot(2,round(ncomponents/2),i)
    vec2word_v1(coeff(:,i));
    title(sprintf('%i',i));
end

%%%% reconstruct using a limited number of Principal Components
ncomponents = 100;
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
ngenerated = 5;
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
%    pause;
end

%%%% sort by Hotelling t squared
[b,i]=sort(tsquared);
for iw=1:200  % nwords
    disp(sprintf('%s %5.1f',orthou{i(iw)},tsquared(i(iw))));
end
for iw=(nwords-200):nwords  % nwords
    disp(sprintf('%s %5.1f',orthou{i(iw)},tsquared(i(iw))));
end
