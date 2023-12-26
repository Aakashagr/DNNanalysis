%%% this version of vectorlex has words of variable length 
%% and also a code for lenght 3-8
%% Pas de loin de Leucade, 14:11 heure grecque (et deux Bakhlava plus loin)

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
ortho = strrep(ortho,'ü','u');
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

minletters = 3;
maxletters = 8;
nletters = maxletters;
nlocations = nletters;

%%% reduce to just the list of words with n letters
listw = find((nblet >= minletters).*(nblet <= maxletters).*(freq>=10).*(badchars==0));
ortho = lower(ortho(listw));
nblet = nblet(listw);
freq = freq(listw);

orthou = unique(ortho); %% 1 exemplar of each word
nwords = length(orthou);

%%% create the letter vector for each word
vectorlex = zeros(nwords,26*nletters + nlocations);
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
    %%% add a code for word length
    vectorlex(iw,26*nletters + length(thisword) )=2;
    %% normalize so that each word vector lies on the unit sphere
    vectorlex(iw,:) = vectorlex(iw,:)/norm(vectorlex(iw,:));
end

figure(1); 
title('Original word codes');
imagesc(vectorlex);
xlabel('letter codes');
ylabel('words');

% show all the word codes
figure(10);
for iw=1:2 %nwords
    clf;
    vec2word_v4(vectorlex(iw,:));
    disp(orthou{iw});
%    pause;
end

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
icomp = 0;
for ii=1:4
    figure(10+ii);
    clf;
    for i =1:ncomponents; %% above 30 the figure gets crowded!
        icomp = icomp +1;
        subplot(2,round(ncomponents/2),i)
        vec2word_v4(coeff(:,icomp));
        title(sprintf('%i',icomp));
    end
end

%%%% reconstruct using a limited number of Principal Components
ncomponents = 50;
reconstructed = (score(:,1:ncomponents) * coeff(:,1:ncomponents)') + repmat(meanvectorlex,nwords,1);

figure(6);
title('Reconstructions of the original words');
imagesc(reconstructed);

%%%% recode the reconstruction as a word
recon_orthou = cell(1,nwords);
for iw = 1:nwords
    recon_orthou{iw} = reconword_v4(reconstructed(iw,:));
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
    random_orthou = reconword_v4(randomword);
    disp(sprintf('%s',random_orthou));
    figure(20);
    clf;
    vec2word_v4(randomword);
    pause;
end
%%% the above does not work so well, because there is no guarantee that it
%%% generates letters at each of the necessary locations
%%% 2nd attempt: generate words with a relatively flat profile of letter
%%% activations
ngenerated = 5;
for iw = 1:ngenerated
    correct = false;
    while ~correct
        coords = randn(1,ncomponents);
        coords = coords / norm(coords);
        randomword = (coords * coeff(:,1:ncomponents)');
        randomword = randomword ./ norm(randomword);
        randomword = randomword + meanvectorlex;
        %%% check that all letters have a strong candidate
        for x=1:nletters
            posvalue(x) = max(randomword(26*(x-1)+(1:26)));
        end
        correct = ( min(posvalue) > 0.2);
    end    
    random_orthou = reconword_v4(randomword);
    disp(sprintf('%s',random_orthou));
    figure(20);
    clf;
    vec2word_v4(randomword);
    pause;
end


%%%% sort by Hotelling t squared
[b,i]=sort(tsquared);
for iw=1:200  % nwords
    disp(sprintf('%s %5.1f',orthou{i(iw)},tsquared(i(iw))));
end
for iw=(nwords-200):nwords  % nwords
    disp(sprintf('%s %5.1f',orthou{i(iw)},tsquared(i(iw))));
end
