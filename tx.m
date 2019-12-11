function t = tx(bits, symb_per, fc, sample_freq, pulse_shape)
%   bits is the input bit sequence (row vector)    
%   symb_per is the symbol period
%   fc is the carrier frequency
%   sample_freq is the sampling frequency
%   pulse_shape is something I don't quite understand yet

%   Set "defaults"
    if ~exist('symb_per', 'var')
        symb_per = 1;
    end
    if ~exist('fc', 'var')
        fc = 1000;
    end
    if ~exist('sample_freq', 'var')
        sample_freq = 48000;
    end    
    if ~exist('pulse_shape', 'var')
        pulse_shape = [1];
    end

%   Split bits into Q and I, make sure they're the same length
    if (mod(size(bits,2),2) == 1)
        xi = bits(1:2:end);
        xq = [bits(2:2:end), 0];
    else
        xi = bits(1:2:end);
        xq = bits(2:2:end);
    end

%   Upsample Q and I to be a good length and set 0s to -1
    xi = upsample(xi,symb_per);
    xq = upsample(xq,symb_per);
    xi(xi==0) = -1;
    xq(xq==0) = -1;
    times = (0:size(xi,2)-1)./sample_freq;
    
%   Modulate the bits
    t = conv(pulse_shape, sqrt(2).*(cos(2.*pi.*fc.*xi.*times) - sin(2.*pi.*fc.*xq.*times)));
    
%   Make a .wav file output
    audiowrite('eatMyAss.wav', t, sample_freq);
end



