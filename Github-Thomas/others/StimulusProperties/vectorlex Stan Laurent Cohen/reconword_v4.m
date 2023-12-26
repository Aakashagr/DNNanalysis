function reconword=reconword_v4(v)
%%%% this function takes a v4 vector and decodes it into a word

nletters = floor(length(v) / 26);
nlocations = mod(length(v),26);

%%% find the length
[m,wordlength] = max(v(26*nletters+(1:nlocations)));

reconword='';
%%% reconstitute the letters
for il= 1:wordlength
    vil = zeros(1,26);
    if il<=4
      vil = vil + v((1:26)+26*(il-1));
    end
    if il>=wordlength-3
      vil = vil + v((1:26)+26*(nletters-1-(wordlength-il)));
    end
    [m,i] = max(vil);
    reconword = [ reconword cast(i+96,'char') ];
end
