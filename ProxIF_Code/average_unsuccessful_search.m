function c_n=average_unsuccessful_search(N)
if N==2
    c_n=1;
elseif N>2
    c_n=2*harmonic(N-1)-2*(N-1)/N;
else
    c_n=0;
end
end