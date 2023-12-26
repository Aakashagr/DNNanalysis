function vectorlex = makevector(thisword,neighborspread);
%%% creates the neural code for any word, assuming a fixed length

nletters = length(thisword);
vectorlex = zeros(1,26 * nletters);
for il = 1:length(thisword)
    thisletter = thisword(il);
    lpos = 26*(il-1)+(cast(thisletter,'int16')-96);
    vectorlex(lpos) = vectorlex(lpos) + 1;
    %%% smear the activation also to neighboring letter positions:
    if il>1
        vectorlex(lpos-26) = vectorlex(lpos-26) + neighborspread;
    end
    if il<nletters
        vectorlex(lpos+26) = vectorlex(lpos+26) + neighborspread;
    end
end
%% normalize so that each word vector lies on the unit sphere
vectorlex = vectorlex/norm(vectorlex);
