function vec2word_v1(v)
%%% takes a vector code v, and plots the corresponding letters
% this function v1 uses the simple fixed n-letter code

nletters = floor(length(v) / 26);
nlocations = mod(length(v),26);

maxabs = max(abs(v));

posvalue = zeros(1,nletters+1);
for x=1:nletters
    for y=1:26
        vvalue = v(26*(x-1)+y);
        posvalue(x) = posvalue(x) + abs(vvalue);
    end
end
x=nletters+1;
for y=1:nlocations
    vvalue = v(26*(x-1)+y);
    posvalue(x) = posvalue(x) + abs(vvalue);
end
%%%% plot how much this vector relies on each letter position
plot(1:nletters+1,28-posvalue);

%%% plot the letters themselves
axis([1 nletters+1 1 28],'ij','off');
for x=1:nletters
    for y=1:26
        vvalue = v(26*(x-1)+y);
        color = [ 1*(vvalue>0) 0 1*(vvalue<0) ];
        letter = cast(96+y,'char');
        text(x,y,letter,'color',color,'HorizontalAlignment','Center','FontSize',1+12*abs(vvalue/maxabs));
    end
end
%%% location code:
x=nletters+1;
for y=1:nlocations
        vvalue = v(26*nletters+y);
        color = [ 1*(vvalue>0) 0 1*(vvalue<0) ];
        letter = sprintf('%i',y);
        text(x,y,letter,'color',color,'HorizontalAlignment','Center','FontSize',1+12*abs(vvalue/maxabs));    
end
