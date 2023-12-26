function vec2word_v1(v)
%%% takes a vector code v, and plots the corresponding letters
% this function v1 uses the simple fixed n-letter code

nletters = round(length(v) / 26);

maxabs = max(abs(v));

axis([1 nletters 1 26],'ij','off');
for x=1:nletters
    for y=1:26
        vvalue = v(26*(x-1)+y);
        color = [ 1*(vvalue>0) 0 1*(vvalue<0) ];
        letter = cast(96+y,'char');
        text(x,y,letter,'color',color,'HorizontalAlignment','Center','FontSize',1+12*abs(vvalue/maxabs));
    end
end